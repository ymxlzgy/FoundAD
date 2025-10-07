import os
import torch
import yaml
import json
import draccus
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import sys
from src.vision_backbone.prismatic.conf import ModelConfig, ModelRegistry
from src.vision_backbone.prismatic.models import get_vision_backbone_and_transform, get_encoder
from src.vision_backbone.prismatic.overwatch import initialize_overwatch
from src.vision_backbone.prismatic.util import set_global_seed

from PIL import Image

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

@dataclass
class Config:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.EXP_DINOSIGLIP_384PX_RESIZE_NAIVE.model_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "no_align" 
    pretrained_checkpoint: Optional[Path] = None                    # TODO
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)

@draccus.wrap()
def init_vit_backbone(cfg: Config):
    torch.cuda.set_device(device=0)
    torch.cuda.empty_cache()

    # hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    # Load Vision Backbone --> on CPU, in Full Precision (initializing model, image_transform via TIMM)
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Create => wraps `vision_backbone`
    cfg.model.arch_specifier = 'none'
    overwatch.info(f"Instantiating ViT backbone for Training Stage = `{cfg.stage}`")
    vit = get_encoder(
        arch_specifier = cfg.model.arch_specifier,
        device=torch.cuda.current_device(),
        vision_backbone=vision_backbone,
    )

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `freeze_backbones()` => Training Stage: `{cfg.stage}`")
    vit.freeze_backbones(cfg.stage)

    # overwatch.info(f"Invoking `load_checkpoint()` => Training Stage: `{cfg.stage}`")
    # vit.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    return vit


from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import PCA
import torch.nn.functional as F
from collections import defaultdict, deque
import torch
import torch.nn as nn

class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected

def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca

def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def plot_feats(image, lr, hr):
    assert len(image.shape) == len(lr.shape) == len(hr.shape) == 3
    seed_everything(0)
    [lr_feats_pca, hr_feats_pca], _ = pca([lr.unsqueeze(0), hr.unsqueeze(0)], dim=9)
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    ax[0, 0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[1, 0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[2, 0].imshow(image.permute(1, 2, 0).detach().cpu())

    ax[0, 0].set_title("Image", fontsize=22)
    ax[0, 1].set_title("Original", fontsize=22)
    ax[0, 2].set_title("Upsampled Features", fontsize=22)

    ax[0, 1].imshow(lr_feats_pca[0, :3].permute(1, 2, 0).detach().cpu())
    ax[0, 0].set_ylabel("PCA Components 1-3", fontsize=22)
    ax[0, 2].imshow(hr_feats_pca[0, :3].permute(1, 2, 0).detach().cpu())

    ax[1, 1].imshow(lr_feats_pca[0, 3:6].permute(1, 2, 0).detach().cpu())
    ax[1, 0].set_ylabel("PCA Components 4-6", fontsize=22)
    ax[1, 2].imshow(hr_feats_pca[0, 3:6].permute(1, 2, 0).detach().cpu())

    ax[2, 1].imshow(lr_feats_pca[0, 6:9].permute(1, 2, 0).detach().cpu())
    ax[2, 0].set_ylabel("PCA Components 7-9", fontsize=22)
    ax[2, 2].imshow(hr_feats_pca[0, 6:9].permute(1, 2, 0).detach().cpu())

    remove_axes(ax)
    plt.tight_layout()
    plt.close(fig)  # Close plt to avoid additional empty plots
    return fig

if __name__ == "__main__":
    import numpy as np

    image_path = "vision_backbone/dog.jpg"
    vit = init_vit_backbone()
    img = Image.open(image_path).convert("RGB")
    img_ori = torch.tensor(np.array(img)).permute(2,0,1)
    feats = vit.generate(img)[0]
    print(feats.shape)
    fig = plot_feats(img_ori, feats[0], feats[0])
    plt.show() 