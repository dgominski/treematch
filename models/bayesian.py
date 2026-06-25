import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Post_Prob(nn.Module):
    """
    Posterior probability P(pixel j | point i) ∝ exp(-dist(i,j)² / 2σ²).

    Softmax is over dim=0 (points), so each pixel's probability mass sums to 1
    across all annotated points (+ optional background).
    """
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        super().__init__()
        assert c_size % stride == 0
        self.sigma    = sigma
        self.bg_ratio = background_ratio
        self.use_bg   = use_background
        self.device   = device
        cood = torch.arange(0, c_size, step=stride, dtype=torch.float32, device=device) + stride / 2
        self.cood    = cood.unsqueeze(0)          # (1, c_size/stride)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, points, st_sizes):
        """
        points   : list[B] of (N_i, 2) float tensors — (row, col) GT coordinates
        st_sizes : list[B] of ints — spatial size per image (used for bg distance)
        Returns  : list[B] of (N_i [+1], H*W) probability tensors, or None if no points
        """
        num_pts  = [len(p) for p in points]
        all_pts  = torch.cat(points, dim=0)        # (ΣN_i, 2)

        if len(all_pts) > 0:
            x = all_pts[:, 0].unsqueeze(1)         # rows  (ΣN_i, 1)
            y = all_pts[:, 1].unsqueeze(1)         # cols  (ΣN_i, 1)
            # squared distance along each axis: (a - b)² = a² - 2ab + b²
            x_dis = -2 * torch.matmul(x, self.cood) + x ** 2 + self.cood ** 2  # (ΣN_i, c_size)
            y_dis = -2 * torch.matmul(y, self.cood) + y ** 2 + self.cood ** 2
            # 2D distance: (ΣN_i, c_size, c_size) → (ΣN_i, H*W)
            dis = (x_dis.unsqueeze(1) + y_dis.unsqueeze(2)).view(len(all_pts), -1)

            prob_list = []
            for dis_i, st_size in zip(torch.split(dis, num_pts), st_sizes):
                if len(dis_i) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(dis_i.min(dim=0, keepdim=True)[0], min=0.0)
                        bg_dis  = (st_size * self.bg_ratio) ** 2 / (min_dis + 1e-5)
                        dis_i   = torch.cat([dis_i, bg_dis], dim=0)
                    prob_list.append(self.softmax(-dis_i / (2.0 * self.sigma ** 2)))
                else:
                    prob_list.append(None)
        else:
            prob_list = [None] * len(points)
        return prob_list


class Bay_Loss(nn.Module):
    """
    Bayesian loss: for each annotated point i, the density map integrated under
    P(·|i) should equal 1.  MAE between that expected count and 1.0.
    """
    def __init__(self, use_background, device):
        super().__init__()
        self.use_bg = use_background
        self.device = device

    def forward(self, prob_list, target_list, pre_density):
        """
        prob_list    : list[B] from Post_Prob — (N_i [+1], H*W) or None
        target_list  : list[B] of (N_i,) float tensors — 1.0 per point
        pre_density  : (B, 1, H, W) predicted density map
        """
        loss = pre_density.new_zeros(1)
        for idx, prob in enumerate(prob_list):
            if prob is None:                              # no annotations → count should be 0
                pre_count = pre_density[idx].sum()
                target    = pre_density.new_zeros(1)
            else:
                N = len(prob)
                if self.use_bg:
                    target       = pre_density.new_zeros(N)
                    target[:-1]  = target_list[idx]       # background target = 0
                else:
                    target = target_list[idx]
                # expected density under posterior for each point: (N [+1],)
                pre_count = (pre_density[idx].view(1, -1) * prob).sum(dim=1)
            loss = loss + torch.abs(target - pre_count).sum()
        return loss / len(prob_list)


class Trainer(object):
    def __init__(self, sigma, device, lr, max_epoch, val_epoch,
                 background_ratio=0.15, use_background=True, **kwargs):
        self.sigma            = sigma
        self.device           = device
        self.lr               = lr
        self.max_epoch        = max_epoch
        self.val_epoch        = val_epoch
        self.background_ratio = background_ratio
        self.use_background   = use_background

    def setup(self, backbone):
        self.device   = torch.device(self.device)
        self.backbone = backbone.to(self.device)

        self.optimizer = optim.AdamW(self.backbone.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)
        self.start_epoch = 0

        self.post_prob = None   # lazy-init on first batch (needs spatial size)
        self.bay_loss  = Bay_Loss(self.use_background, self.device)

    def _ensure_post_prob(self, H, W):
        assert H == W, "Post_Prob requires square inputs"
        if self.post_prob is None:
            self.post_prob = Post_Prob(
                sigma=self.sigma,
                c_size=H,
                stride=1,
                background_ratio=self.background_ratio,
                use_background=self.use_background,
                device=self.device,
            )

    def train_step(self, inputs, valid, gt_discrete, logger):
        inputs = inputs.to(self.device)
        valid  = inputs[:, [-1,]].to(self.device)
        N, _, H, W = inputs.size()

        self._ensure_post_prob(H, W)

        points_list = []
        target_list = []
        for b in range(N):
            pts = torch.nonzero(gt_discrete[b, 0], as_tuple=False).float().to(self.device)
            points_list.append(pts)
            target_list.append(torch.ones(len(pts), dtype=torch.float32, device=self.device))

        with torch.set_grad_enabled(True):
            density   = F.softplus(self.backbone(inputs)) * valid   # (N, 1, H, W)
            prob_list = self.post_prob(points_list, [H] * N)
            loss      = self.bay_loss(prob_list, target_list, density)

            gt_counts  = torch.tensor([len(p) for p in points_list], dtype=torch.float32)
            pre_counts = density.view(N, -1).sum(dim=1).detach().cpu()
            mae        = torch.abs(pre_counts - gt_counts).mean().item()

            logger.log({'train/loss': loss.item(), 'train/mae': mae})
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, inputs):
        inputs = inputs.to(self.device)
        valid  = inputs[:, [-1,]].to(self.device)
        with torch.no_grad():
            density = F.softplus(self.backbone(inputs))
        return density * valid   # sums to ~tree_count (no /100 — Bay_Loss calibrates directly)

    def train(self):
        self.backbone.train()

    def eval(self):
        self.backbone.eval()
