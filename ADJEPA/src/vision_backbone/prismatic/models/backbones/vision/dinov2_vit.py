"""
dinov2_vit.py
"""

from src.vision_backbone.prismatic.models.backbones.vision.base_vision import TimmViTBackbone
import numpy as np

# Registry =>> Supported DINOv2 Vision Backbones (from TIMM) =>> Note:: Using DINOv2 w/ Registers!
#   => Reference: https://arxiv.org/abs/2309.16588
DINOv2_VISION_BACKBONES = {"dinov2-vit-l": "vit_large_patch14_reg4_dinov2.lvd142m"}


class DinoV2ViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, reshape=False) -> None:
        super().__init__(
            vision_backbone_id,
            DINOv2_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )
        self.reshape = reshape

    def forward(self, pixel_values):
        feat = super().forward(pixel_values)[0]
        if self.reshape:
            B, N, D = feat.shape
            feat = feat.view(-1, int(np.sqrt(N)), int(np.sqrt(N)), D).permute(0, 3, 1, 2)

        return feat
