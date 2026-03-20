import torch
import os
from data.ps import mean as ps_mean, std as ps_std
from data.gf import mean as gf_mean, std as gf_std
from data.spot import mean as spot_mean, std as spot_std
from torchvision.transforms.functional import normalize


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_image_fps(target_dir, ext=None):
    input_images = []

    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if ext is not None:
                if file.endswith(ext):
                    input_images.append(os.path.join(root, file))
            else:
                if file.endswith((".tif", ".tiff", ".jp2")):
                    input_images.append(os.path.join(root, file))

    print(f"Found {len(input_images)} valid images in {target_dir}.")

    return sorted(input_images)


def split_tensor(tensor, patch_size=256, overlap=64, pad_mode="reflect"):
    """
    Splits a tensor into a batch of overlapping patches, with zero padding to maintain the same patch size.
    Returns all parameters needed to reconstruct the tensor later:
        - mask_p: for each patch gives the area that was originally covered by actual data (not padding)
        - original shape of the tensor
        - x and y padding
    """
    stride = patch_size - overlap
    # compute padding to cover whole image
    nb_windows_height = 1 + (tensor.shape[2] - patch_size - 1) // stride
    nb_windows_width = 1 + (tensor.shape[3] - patch_size - 1) // stride
    p0 = 0
    p1 = 0
    if nb_windows_height * stride + patch_size != tensor.shape[2]:
        p0 = int((1 + patch_size - tensor.shape[2] + stride * nb_windows_height) / 2)
    if nb_windows_width * stride + patch_size != tensor.shape[3]:
        p1 = int((1 + patch_size - tensor.shape[3] + stride * nb_windows_width) / 2)

    padded_image = torch.nn.functional.pad(tensor, pad=(p1, p1, p0, p0), mode=pad_mode)
    unfold = torch.nn.Unfold(kernel_size=(patch_size, patch_size), stride=stride, dilation=1)
    patches = unfold(padded_image)

    patches = patches.reshape(tensor.shape[1], patch_size, patch_size, -1).permute(3, 0, 1, 2)

    # Compute the offsets of each patch in the original image
    i = torch.arange(nb_windows_height + 1, dtype=torch.int) * stride - p0
    j = torch.arange(nb_windows_width + 1, dtype=torch.int) * stride - p1
    offsets = torch.stack(torch.meshgrid(i, j, indexing="ij"), dim=-1)
    offsets = offsets.reshape(patches.shape[0], 2)

    return patches, (tensor.size(2), tensor.size(3)), p0, p1, offsets


def rebuild_tensor(patches, t_size, overlap, p0, p1):
    """
    Rebuilds a tensor that was split in overlapping patches. Overlapping areas are averaged.
    Needs some important arguments to properly reconstruct:
        - original shape of the tensor
        - x and y padding
    """
    tile_size = patches.shape[2]
    stride = tile_size - overlap

    patches = patches.permute(1, 2, 3, 0).reshape(patches.shape[1], patches.shape[2]*patches.shape[3], -1)
    fold = torch.nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride, padding=[p0, p1])
    reconstructed_tensor = fold(patches)
    return reconstructed_tensor[:, [0]]


def normalize_ps(image):
    image = normalize(image, ps_mean, ps_std)
    return image


def denormalize_ps(image):
    image = image[:4] * torch.Tensor(ps_std)[:, None, None] + torch.Tensor(ps_mean)[:, None, None]
    # 2️⃣ Percentile clipping (per channel)
    flat = image.flatten(0)
    lo = torch.quantile(flat, 2 / 100, dim=0, keepdim=True)
    hi = torch.quantile(flat, 98 / 100, dim=0, keepdim=True)
    lo = lo[:, None, None]
    hi = hi[:, None, None]
    image = image.clamp(lo, hi)
    image = (image - lo) / (hi - lo + 1e-8)
    image = image * 255.0
    image = image * 1.8
    return image[[2, 1, 0]].clamp(0, 255).round().to(torch.uint8)

def normalize_gf(image):
    image = normalize(image, gf_mean, gf_std)
    return image

def denormalize_gf(image):
    image = image[:4] * torch.Tensor(gf_std)[:, None, None] + torch.Tensor(gf_mean)[:, None, None]
    # 2️⃣ Percentile clipping (per channel)
    flat = image.flatten(0)
    lo = torch.quantile(flat, 2 / 100, dim=0, keepdim=True)
    hi = torch.quantile(flat, 98 / 100, dim=0, keepdim=True)
    lo = lo[:, None, None]
    hi = hi[:, None, None]
    image = image.clamp(lo, hi)
    image = (image - lo) / (hi - lo + 1e-8)
    image = image * 255.0
    image = image * 1.2
    return image.clamp(0, 255).round().to(torch.uint8)


def denormalize_spot(image):
    image = image[:4] * torch.Tensor(spot_std)[:, None, None] + torch.Tensor(spot_mean)[:, None, None]
    # # 2️⃣ Percentile clipping (per channel)
    # flat = image.flatten(0)
    # lo = torch.quantile(flat, 2 / 100, dim=0, keepdim=True)
    # hi = torch.quantile(flat, 98 / 100, dim=0, keepdim=True)
    # lo = lo[:, None, None]
    # hi = hi[:, None, None]
    # image = image.clamp(lo, hi)
    # image = (image - lo) / (hi - lo + 1e-8)
    # image = image * 255.0
    image = image * 2
    return image.clamp(0, 255).round().to(torch.uint8)


def scale_ls_frame(frame):
    # normalize to mean 0, std 1
    image = (frame - LS_MEAN.reshape(7, 1, 1)) / LS_STD.reshape(7, 1, 1)
    return image