import torch
from torch.nn import Module
import torchvision.models as tmodels
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import datetime
from utils import AverageMeter
import time
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp


M_EPS = 1e-16

class Trainer(object):
    def __init__(self, imsize, device, wot, wtv, num_of_iter_in_ot, reg, lr, max_epoch, **kwargs):
        self.imsize = imsize
        self.device = device
        self.wot = wot
        self.wtv = wtv
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg
        self.lr = lr
        self.max_epoch = max_epoch

    def setup(self, backbone):
        self.device = torch.device(self.device)

        self.backbone = backbone.to(self.device)

        self.optimizer = optim.AdamW(self.backbone.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)

        self.start_epoch = 0

        self.ot_loss = OT_Loss(self.imsize, 1, self.device, self.num_of_iter_in_ot, self.reg)
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mse = nn.MSELoss().to(self.device)
        self.mae = nn.L1Loss().to(self.device)

    def train_step(self, inputs, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)

        #convert gt_discrete to points
        points = []
        for b in range(gt_discrete.size(0)):
            inds = torch.nonzero(gt_discrete[b, 0, :, :], as_tuple=False)
            points.append(inds.float())
        gd_count = np.array([len(p) for p in points], dtype=np.float32)
        points = [p.to(self.device) for p in points]
        gt_discrete = gt_discrete.to(self.device)
        N = inputs.size(0)

        down_h = gt_discrete.size(2)
        down_w = gt_discrete.size(3)
        gt_discrete = gt_discrete.reshape([gt_discrete.shape[0], 1, down_h, 1, down_w, 1]).sum(axis=(3, 5))

        with torch.set_grad_enabled(True):
            outputs = nn.functional.relu(self.backbone(inputs))
            outputs = outputs * valid  # mask invalid regions
            outputs_normed = outputs / (outputs.sum(1).sum(1).sum(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) + 1e-6)

            # Compute OT loss.
            ot_loss, wd, ot_obj_value = self.ot_loss(outputs_normed, outputs, points)
            ot_loss = ot_loss * self.wot
            ot_obj_value = ot_obj_value * self.wot

            # Compute counting loss.
            count_loss = self.mae(outputs.sum(1).sum(1).sum(1),
                                  torch.from_numpy(gd_count).float().to(self.device))

            # Compute TV loss.
            gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            gt_discrete_normed = gt_discrete / (gd_count_tensor + 1e-6)
            tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed).sum(1).sum(1).sum(1) * torch.from_numpy(gd_count).float().to(self.device)).mean(0) * self.wtv

            pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
            mae = np.mean(np.abs(pred_count - gd_count))

            loss = count_loss + tv_loss + ot_loss
            logger.log({
                'train/total_loss': loss.item(),
                'train/ot_loss': ot_loss.item(),
                'train/ot_obj_value': ot_obj_value.item(),
                'train/wasserstein_distance': wd,
                'train/count_loss': count_loss.item(),
                'train/tv_loss': tv_loss.item(),
                'train/mae': mae,
            })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        valid = inputs[:, [-1,]].to(self.device)
        with torch.no_grad():
            outputs = nn.functional.relu(self.backbone(inputs)) * valid
        return outputs

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()


def sinkhorn(a, b, C, reg=1e-1, method='sinkhorn', maxIter=1000, tau=1e3,
             stopThr=1e-9, verbose=False, log=True, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    method : str
        method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
    # elif method.lower() == 'sinkhorn_stabilized':
    #     return sinkhorn_stabilized(a, b, C, reg, maxIter=maxIter, tau=tau,
    #                                stopThr=stopThr, verbose=verbose, log=log,
    #                                warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
    #                                **kwargs)
    # elif method.lower() == 'sinkhorn_epsilon_scaling':
    #     return sinkhorn_epsilon_scaling(a, b, C, reg,
    #                                     maxIter=maxIter, maxInnerIter=100, tau=tau,
    #                                     scaling_base=0.75, scaling_coef=None, stopThr=stopThr,
    #                                     verbose=False, log=log, warm_start=warm_start, eval_freq=eval_freq,
    #                                     print_freq=print_freq, **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # # allocate memory beforehand
    # KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    # Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        KTu = torch.matmul(u, K)
        v = torch.div(b, KTu + M_EPS)
        Kv = torch.matmul(K, v)
        u = torch.div(a, Kv + M_EPS)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = reg * torch.log(u + M_EPS)
        log['beta'] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P


class OT_Loss(Module):
    def __init__(self, c_size, stride, device, num_of_iter_in_ot=100, reg=10.0):
        super(OT_Loss, self).__init__()
        assert c_size % stride == 0

        self.c_size = c_size
        self.device = device
        self.num_of_iter_in_ot = num_of_iter_in_ot
        self.reg = reg

        # coordinate is same to image space, set to constant since crop size is same
        coords = torch.arange(0, c_size, step=stride, dtype=torch.float32, device=device) + stride / 2
        # Build full 2D coordinate grids (X and Y), then flatten to shape [1, H*W]
        X, Y = torch.meshgrid(coords, coords, indexing="xy")  # X: (H, W), Y: (H, W)
        self.coord_x = X.reshape(1, -1).to(device)  # [1, H*W]
        self.coord_y = Y.reshape(1, -1).to(device)  # [1, H*W]

        self.density_size = coords.numel()

    def forward(self, normed_density, unnormed_density, points):
        batch_size = normed_density.size(0)
        assert len(points) == batch_size
        assert self.density_size == normed_density.size(2)
        loss = torch.zeros([1]).to(self.device)
        ot_obj_values = torch.zeros([1]).to(self.device)
        wd = 0 # wasserstain distance
        for idx, im_points in enumerate(points):
            if len(im_points) > 0:
                # compute l2 square distance, it should be source target distance. [#gt, #cood * #cood]
                # avoid in-place modification of passed im_points
                im_pts = im_points.clone().to(self.device).float()

                # im_pts: [#gt, 2], assumed format (x, y) compatible with X,Y indexing="xy"
                x = im_pts[:, 0].unsqueeze(1)  # [#gt, 1]
                y = im_pts[:, 1].unsqueeze(1)  # [#gt, 1]

                # compute 2D squared euclidean distances: broadcast properly
                # x_dis: [#gt, H*W], y_dis: [#gt, H*W]
                x_dis = (x - self.coord_x) ** 2
                y_dis = (y - self.coord_y) ** 2

                dis = x_dis + y_dis

                source_prob = normed_density[idx][0].view([-1]).detach()
                target_prob = (torch.ones([len(im_points)]) / len(im_points)).to(self.device)
                # use sinkhorn to solve OT, compute optimal beta.
                P, log = sinkhorn(target_prob, source_prob, dis, self.reg, maxIter=self.num_of_iter_in_ot, log=True)

                loss += torch.sum(dis * P)
                wd += torch.sum(dis * P).item()

                # #
                # H = self.density_size
                # W = self.density_size
                # src_density_img = normed_density[idx][0].detach().cpu().view(H, W)
                # P_img = P.detach().cpu().view(len(im_points), H, W)
                # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                # axes[0].imshow(src_density_img, cmap="viridis")
                # axes[0].set_title("Source density (predicted)")
                # axes[0].scatter(
                #     im_points[:, 0].cpu(),
                #     im_points[:, 1].cpu(),
                #     s=10, c='red'
                # )
                # axes[0].invert_yaxis()
                # transport_map = P_img.sum(dim=0)
                # axes[1].imshow(transport_map, cmap="magma", origin='lower')
                # axes[1].set_title("Aggregated OT transport map\n(sum of P over targets)")
                # example_id = 0
                # axes[2].imshow(P_img[example_id], cmap="plasma", origin='lower')
                # axes[2].set_title(f"OT map for target point #{example_id}")
                # plt.tight_layout()
                # plt.show()

        return loss, wd, ot_obj_values




if __name__ == "__main__":
    # test OT
    ot_loss = OT_Loss(32, 1, torch.device("cpu"), 100, 10.0)

    dummy_pred = torch.zeros([2, 1, 32, 32])
    points = [torch.tensor([[5.0, 5.0], [10.0, 10.0], [14.0, 14.0]]), torch.tensor([[15.0, 15.0]])]
    dummy_pred[0, 0, 6, 8] = 1.0
    dummy_pred[0, 0, 10, 10] = 1.0
    pred_normed = dummy_pred / (dummy_pred.sum(1).sum(1).sum(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) + 1e-6)
    ot_loss, wd, ot_obj_value = ot_loss(pred_normed, dummy_pred, points)
    print(ot_loss, wd, ot_obj_value)
