from __future__ import annotations
import sys
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra


def load_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # noqa
        except ModuleNotFoundError:
            sys.exit("PyYAML is required for YAML configs. Install via `pip install pyyaml`.")
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    else:
        return json.loads(path.read_text())


def sample_images(
    source_root: Path,
    target_root: Path,
    num_samples: int,
    train_subpaths: Tuple[str, ...],
    allowed_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    rename_images: bool = True,
) -> None:
    allowed_exts = tuple(ext.lower() for ext in allowed_exts)

    target_train_root = target_root / "train"
    target_train_root.mkdir(parents=True, exist_ok=True)

    for category_dir in filter(Path.is_dir, source_root.iterdir()):
        cat_name = category_dir.name

        imgs: List[Path] = []
        chosen_subpath = None
        for sub in train_subpaths:
            candidate = category_dir / sub
            if candidate.is_dir():
                found = [
                    p for p in candidate.iterdir()
                    if p.is_file() and p.suffix.lower() in allowed_exts
                ]
                if found:
                    imgs = found
                    chosen_subpath = sub
                    break

        if not imgs:
            print(f"[skip] {cat_name}: none of {list(train_subpaths)} contained images")
            continue

        k = min(num_samples, len(imgs))
        # selected = random.sample(imgs, k)
        random.shuffle(imgs)
        selected = imgs[:k]

        dest_dir = target_train_root / cat_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        existing_files = [
            p for p in dest_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed_exts
        ]
        start_idx = len(existing_files)

        for i, src in enumerate(selected):
            new_name = (f"{start_idx + i:03d}{src.suffix.lower()}") if rename_images else src.name
            shutil.copy2(src, dest_dir / new_name)

        print(f"[✓] {cat_name}: copied {k}/{len(imgs)} from '{chosen_subpath}'")



@dataclass
class SamplerCfg:
    source: str = ""          # Must（CLI covered）
    target: str = ""          # Must（CLI covered）

    num_samples: int = 1
    train_subpaths: List[str] = field(default_factory=lambda: ["train/good", "train/ok"])
    allowed_exts: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    rename_images: bool = True

    seed: int | None = None

    user_config: str = ""


def _finalize_cfg(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.user_config:
        ext = load_config_file(Path(cfg.user_config))
        cfg = OmegaConf.merge(OmegaConf.structured(SamplerCfg), cfg, ext)

    source = cfg.get("source")
    target = cfg.get("target")
    if not source or not target:
        sys.exit("Both 'source' and 'target' must be specified. "
                 "Provide via CLI (source=..., target=...) or user_config=...")

    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_few_shot")
def main(cfg: DictConfig) -> None:
    cfgd = _finalize_cfg(cfg)
    random.seed(int(cfgd["seed"]))

    sample_images(
        source_root=Path(cfgd["source"]),
        target_root=Path(cfgd["target"]),
        num_samples=int(cfgd["num_samples"]),
        train_subpaths=tuple(cfgd["train_subpaths"]),
        allowed_exts=tuple(cfgd["allowed_exts"]),
        rename_images=bool(cfgd["rename_images"]),
    )


if __name__ == "__main__":
    main()