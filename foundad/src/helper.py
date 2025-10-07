import logging
from pathlib import Path
from typing import Any, Dict, List
import logging
import sys
import math
import torch

import numpy as np
import torch
from matplotlib import cm, pyplot as plt
from PIL import Image


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def save_segmentation_grid(
    save_dir: Path,
    image_names: List[str],
    images: np.ndarray,
    masks: np.ndarray,
    segs: np.ndarray,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, img, mask, seg in zip(image_names, images, masks, segs):
        if img.dtype != np.uint8:
            img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img
        seg_norm = (seg - seg.min()) / (seg.max() - seg.min() + 1e-8)
        seg_rgb = (cm.jet(seg_norm)[..., :3] * 255).astype(np.uint8)
        safe_name = name.replace("/", "_") + ".png"
        out_path = save_dir / safe_name

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(img_uint8);  axs[0].set_title("Original");  axs[0].axis("off")
        axs[1].imshow(mask, cmap="gray");     axs[1].set_title("GT mask");   axs[1].axis("off")
        axs[2].imshow(seg_rgb);               axs[2].set_title("Heatmap");   axs[2].axis("off")
        overlay = Image.blend(
            Image.fromarray(img_uint8).convert("RGB"),
            Image.fromarray(seg_rgb).convert("RGB"),
            alpha=0.5,
        )
        axs[3].imshow(overlay);               axs[3].set_title("Overlay");   axs[3].axis("off")
        fig.tight_layout(); fig.savefig(out_path); plt.close(fig)




def init_opt(
    predictor,
    wd=1e-4,
    lr=0.005,
    lr_config='const',
    use_bfloat16=False,
    max_epoch=2000,       # for cosine_warmup
    min_lr=1e-6,          # for cosine_warmup
    warmup_epoch=5,       # for cosine_warmup
    step_size=300,         # for step
    gamma=0.1,            # for step
):
    param_groups = [
        {
            'params': list(predictor.parameters()),
            'lr': lr,           
            'weight_decay': wd      
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)  

    if lr_config == 'const':
        logger.info(f'Constant lr, lr={lr}')
        scheduler = None
    elif lr_config == 'step':
        logger.info(f'Step lr: size={step_size},gamma={gamma}')
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif lr_config == 'cosine':
        logger.info(f'Consine Anneal lr, min_lr={min_lr}')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epoch, eta_min=min_lr
        )
    elif lr_config == 'cosine_warmup':
        logger.info(f'Consine Warmup lr, warmup_epoch={warmup_epoch}, min_lr={min_lr}')
        def lr_lambda(epoch):
            if epoch < warmup_epoch:
                # 0 → 1
                return float(epoch + 1) / float(warmup_epoch)
            # cosine
            progress = (epoch - warmup_epoch) / float(max_epoch - warmup_epoch)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # [1 → 0]
            return (min_lr / lr) + (1 - min_lr / lr) * cosine  # [1 → eta_min/lr]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown lr_config: {lr_config}")
    
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

    return optimizer, scheduler, scaler