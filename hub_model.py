"""
Self-contained UNetR50 for tree density estimation.

Dependencies: torch, segmentation-models-pytorch, huggingface_hub

Load a pretrained checkpoint:
    from hub_model import UNetR50
    model = UNetR50.from_pretrained("dgominski/TinyTrees", subfolder="ps")   # Rwanda
    model = UNetR50.from_pretrained("dgominski/TinyTrees", subfolder="gf")   # China
    model = UNetR50.from_pretrained("dgominski/TinyTrees", subfolder="spot") # France

Forward pass:
    # x: (B, in_channels+1, H, W) — image bands + binary validity mask (last channel)
    density_map = model(x)          # (B, 1, H, W), trees/pixel
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub import constants as hf_constants


class UNetR50(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="treedensity",
    repo_url="https://huggingface.co/dgominski/TinyTrees",
    license="apache-2.0",
    coders={},
):
    """ResNet-50 U-Net for per-pixel tree density regression.

    Args:
        in_channels: number of image bands (4 for PS/GF/SPOT). The model
            expects in_channels+1 actual input channels: the image bands
            concatenated with a binary validity mask as the last channel.
    """

    def __init__(self, in_channels: int = 4, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.unet = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=in_channels + 1,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    def get_feats(self, x: torch.Tensor) -> torch.Tensor:
        """Return decoder feature map (used by P2PNet head)."""
        return self.unet.decoder(self.unet.encoder(x))

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        subfolder: Optional[str] = None,
        **model_kwargs,
    ):
        model = cls(**model_kwargs)

        def _dl(filename):
            path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
            return hf_hub_download(
                repo_id=model_id,
                filename=path_in_repo,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        if os.path.isdir(model_id):
            base = os.path.join(model_id, subfolder) if subfolder else model_id
            sf = os.path.join(base, hf_constants.SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, sf, map_location, strict)

        try:
            return cls._load_as_safetensor(
                model, _dl(hf_constants.SAFETENSORS_SINGLE_FILE), map_location, strict
            )
        except Exception:
            return cls._load_as_pickle(
                model, _dl(hf_constants.PYTORCH_WEIGHTS_NAME), map_location, strict
            )
