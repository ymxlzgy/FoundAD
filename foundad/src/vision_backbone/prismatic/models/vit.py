"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.vision_backbone.prismatic.models.backbones.llm import LLMBackbone
from src.vision_backbone.prismatic.models.backbones.llm.prompting import PromptBuilder
from src.vision_backbone.prismatic.models.backbones.vision import VisionBackbone
from src.vision_backbone.prismatic.models.vlms.base_vlm import VLM
from src.vision_backbone.prismatic.overwatch import initialize_overwatch
from src.vision_backbone.prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class VitEncoder():
    def __init__(
        self,
        vision_backbone: VisionBackbone,
        device,
        arch_specifier: str = "gelu-mlp",
    ) -> None:
        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        self.device = device
        self.vision_backbone = vision_backbone.to(self.device)
        

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, vision_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, vision_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, vision_backbone.embed_dim)
        elif arch_specifier.endswith("none"):
            self.projector = torch.nn.Identity()
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "projector"]
        self.trainable_module_keys = []

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        vision_backbone: VisionBackbone,
        arch_specifier: str = "gelu-mlp",
    ) -> VitEncoder:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vit = cls(
            vision_backbone,
            arch_specifier=arch_specifier,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict
        ), "ViT `from_pretrained` expects checkpoint with keys for `projector`!"

        vit.projector.load_state_dict(model_state_dict["projector"])

        # Freeze Weights
        vit.requires_grad_(False)
        vit.eval()

        return vit

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "finetune":
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "no_align":
            self.vision_backbone.requires_grad_(False)

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info("No Projector", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    # def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
    #     """Load weights from checkpoint (if required by the given stage)."""
    #     assert stage in {"align", "finetune", "no_align"}, f"Stage {stage} is not supported!"

    #     # If we're running a `none` architecture, we're good!
    #     if self.arch_specifier.startswith("none"):
    #         overwatch.info(
    #             f"VitEncoder with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
    #         )
    #         return

    #     # Otherwise, handle stage-specific logic!
    #     if stage == "align":
    #         overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
    #         return

    #     # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
    #     overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

    #     # Config specifies path to a checkpoint to load
    #     if pretrained_checkpoint is not None:
    #         overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
    #         model_state_dict = torch.load(pretrained_checkpoint)["model"]
    #         self.projector.load_state_dict(model_state_dict["projector"])

    #         return

    #     # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
    #     model, scale, _, seed = run_dir.name.split("+")
    #     align_dirs = [
    #         d
    #         for d in run_dir.parent.iterdir()
    #         if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
    #     ]
    #     assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
    #     if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
    #         overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
    #         model_state_dict = torch.load(pretrained_checkpoint)["model"]
    #         self.projector.load_state_dict(model_state_dict["projector"])
    #     else:
    #         raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def forward(
        self,
        pixel_values,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        
        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k].float() for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values.float())

        # Projection Logic :: [bsz, num_patches, embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        return projected_patch_embeddings, projected_patch_attention_mask

    @torch.inference_mode()
    def generate(self, image: Image, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform = self.vision_backbone.image_transform

        # Prepare Inputs
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        feats = self.forward(pixel_values)

        return feats
