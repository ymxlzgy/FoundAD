
import multiprocessing as mp
from typing import Any, Dict, Tuple, Optional, List
import importlib   
import yaml, numpy as np, torch
import torch.nn as nn
from PIL import Image

from src.utils.tensors import trunc_normal_
from src.datasets.dataset import build_dataloader
import src.dinov2.models.vision_transformer as vit
from transformers import AutoProcessor, SiglipVisionModel, CLIPVisionModel



class LinearProjector(torch.nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class VisionModule(nn.Module):
    def __init__(self, model_name: str, pred_depth: int, pred_emb_dim: int, use_cuda: bool = True, if_pe: bool = True):
        super().__init__()
        (self.encoder, self.num_patches, self.embed_dim, self.processor, self.projector) = self._build_encoder(model_name)
        self.model_name = model_name

        self.predictor = vit.__dict__["vit_predictor"](num_patches=self.num_patches, embed_dim=self.embed_dim,
                                                         predictor_embed_dim=pred_emb_dim, depth=pred_depth,if_pe=if_pe)
        self._init_predictor(self.predictor)
        self.dropout = nn.Dropout(0.2)
        if use_cuda and torch.cuda.is_available():
            self.cuda()

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)
    
    def target_features(self, images, paths):
        with torch.no_grad():
            return self._extract(images, paths)

    def context_features(self, images, paths):
        z = self._extract(images, paths)
        p = self.predictor(self.dropout(z))
        return z, p

    def _build_encoder(self, model: str):

        projector = processor = None
        if model == "dinov2":
            enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval(); num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dinov3":
            enc = torch.hub.load("facebookresearch/dinov3", 'dinov3_vitb16', weights='https://dinov3.llamameta.net/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoicjdweHhiNDNjaG51cmNpODdqcDVrOG5mIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0YS5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE3NTU0MzcxMzh9fX1dfQ__&Signature=FKi6FewJE1uh80XjrqFrZzNkwnwvhPaAMXqEhO5F9gcBnX4j-lN2ejRIo2Pfj7ccGUyb2he19F8J43RGeA0bHuUKK%7EDjMK-y6tTab3r8HMxGeYtqSYFw5DGt1PnU4rPvRDdSgXgjKD%7E02pW04WtTSPyGmXsV6cXHM1rsNzryA%7EOP1mXdC5h93PsfplRJjkxVJhRrIEKGmQS5CDnUZqd%7Equjx3ENLW%7Ej2ybAXYuaoTkXC1pEwLVaZ2GdUevStXFV9h3b7D-dsvqWhuE-46dOCPKtyfTGaCE3h8Sj%7E7wBZ2NQMBU3EX9fI9emOi81FSOwK5BbQqd7AQ73QqKMGm9JzLQ__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1110542151137970').eval()
            num_patches, embed_dim = enc.patch_embed.num_patches, enc.embed_dim
        elif model == "dino":
            enc = torch.hub.load("facebookresearch/dino:main", "dino_vitb16").eval(); num_patches, embed_dim = 1024, enc.embed_dim
        elif model == "siglip":
            enc = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-512").eval(); processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-512"); num_patches, embed_dim = 1024, 768
        elif model == "clip":
            enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16").eval(); processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16"); num_patches, embed_dim = 196, 768
        elif model == "dinosiglip":
            from src.vision_backbone.scripts.vit_inference import init_vit_backbone, Config      
            
            config = Config()
            enc = init_vit_backbone(config)

            projector = LinearProjector(2176, 2176).cuda()
            num_patches, embed_dim = 729, 2176
        else:
            raise ValueError(f"Unknown model: {model}")
        if model != 'dinosiglip':
            for p in enc.parameters(): 
                p.requires_grad = False
        return enc, num_patches, embed_dim, processor, projector

    def _extract(self, imgs: torch.Tensor, paths: List[str]):
        if self.model_name == "dinov2":
            h = self.encoder.get_intermediate_layers(imgs, n=3, return_class_token=False)[0] # the thrid last block
        elif self.model_name == "dinov3":
            h = self.encoder.get_intermediate_layers(imgs, n=3, return_class_token=False)[0] 
        elif self.model_name == "dino":
            h = self.encoder.get_intermediate_layers(imgs, n=3)[0][:,1:,:]
        elif self.model_name == "siglip":
            feats = [self.encoder(**self.processor(Image.open(p).convert("RGB"), return_tensors="pt").to(imgs.device)).last_hidden_state for p in paths]
            h = torch.cat(feats, dim=0)
        elif self.model_name == "clip":
            h = self.encoder(imgs).last_hidden_state[:,1:,:]
        elif self.model_name == "dinosiglip":
            feats = [self.encoder.generate(Image.open(p).convert("RGB"))[0] for p in paths]
            h = torch.cat(feats).view(imgs.size(0), 2176, -1).permute(0,2,1)
            h = self.projector(h) if self.projector else h
        else:
            raise NotImplementedError(self.model_name)
        
        # print(imgs.shape)
        # print(h.shape)

        return h

    @staticmethod
    def _init_predictor(module):
        for m in module.modules():
            if isinstance(m, nn.Linear): trunc_normal_(m.weight, std=0.02); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm): nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0)
