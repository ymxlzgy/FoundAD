import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm, pyplot as plt
from PIL import Image

from src.datasets.dataset import build_dataloader
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.ADJEPA import VisionModule          

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")


def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
    ).eval()

@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    logger.info("Evaluating %s", ckpt.name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"]).to(device)
    state = torch.load(ckpt, map_location=device)
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])

    crop = cfg["meta"]["crop_size"]
    K = cfg["testing"]["K_top"]
    classnames = cfg["data"]["classnames"]
    dataset_name = cfg["data"].get("dataset", "mvtec")

    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "pix_auroc"),  ("%.8f", "pro_auc"),
    )

    inst_auc, inst_aupr, pix_auc, pro_auc = [], [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in classnames:
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["img_path"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
        )

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []

        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

            enc = model.target_features(img, paths)
            pred = model.predict(enc)
            l2 = F.mse_loss(enc, pred, reduction="none").mean(dim=2)      

            topk = torch.topk(l2, K, dim=1).values.mean(dim=1)
            patch_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l2.size(1)))
            pix = F.interpolate(l2.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu()); img_buf.append(img.cpu()); mask_buf.append(mask.cpu())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8)

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro  = calculate_pro(mask_np, pix_norm,
                             max_steps=cfg["testing"]["max_steps"], expect_fpr=cfg["testing"]["expect_fpr"])

        logger.info("%s | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                    cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)
        csv_logger.log(ckpt.name, cls, inst["auroc"], inst["aupr"], pix["auroc"], pro)

        inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
        pix_auc.append(pix["auroc"]);   pro_auc.append(pro)

        # Generate visualizations
        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("Mean | AUROC_i %.4f | AUPR_i %.4f | AUROC_p %.4f | PRO-AUC %.4f",
                np.mean(inst_auc), np.mean(inst_aupr), np.mean(pix_auc), np.mean(pro_auc))
    csv_logger.log(ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
                   np.mean(pix_auc), np.mean(pro_auc))

def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["checkpoint_path"])
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()