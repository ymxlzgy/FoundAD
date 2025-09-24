
import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  
        except ModuleNotFoundError as exc: 
            sys.exit(
                "PyYAML is required for YAML configs. "
                "Install it via `pip install pyyaml`."
            )
        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    else:  
        with path.open("r") as f:
            return json.load(f)


def sample_images(
    source_root: Path,
    target_root: Path,
    num_samples: int,
    allowed_exts: Tuple[str, ...] = (
        ".png",
        ".jpg",
    ),
    rename_images: bool = True,
) -> None:
    target_train_root = target_root / "train"
    target_train_root.mkdir(parents=True, exist_ok=True)
    train_subpaths = ["train/good", "train/ok"]


    for category_dir in filter(Path.is_dir, source_root.iterdir()):
        cat_name = category_dir.name
        imgs: List[Path] = []

        chosen_subpath = None
        for sub in train_subpaths:
            candidate = category_dir / sub
            if candidate.is_dir():
                imgs = [
                    p
                    for p in candidate.iterdir()
                    if p.is_file() and p.suffix.lower() in allowed_exts
                ]
                if imgs:
                    chosen_subpath = sub
                    break

        if not imgs:
            print(f"[skip] {cat_name}: none of {train_subpaths} contained images")
            continue

        k = min(num_samples, len(imgs))
        selected = random.sample(imgs, k)

        dest_dir = target_train_root / cat_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        existing_files = [
            p for p in dest_dir.iterdir() if p.is_file() and p.suffix in allowed_exts
        ]
        start_idx = len(existing_files)

        for i, src in enumerate(selected):
            new_name = (
                f"{start_idx + i:03d}{src.suffix.lower()}"
                if rename_images
                else src.name
            )
            shutil.copy2(src, dest_dir / new_name)

        print(f"[✓] {cat_name}: copied {k}/{len(imgs)} from '{chosen_subpath}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Few‑shot sampler for MVTec‑style datasets"
    )
    parser.add_argument("--config", type=Path, help="Path to JSON or YAML config file")

    parser.add_argument("--source", type=Path, help="Dataset root")
    parser.add_argument("--target", type=Path, help="Output root for few‑shot data")
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        help="Images per category (overrides config)",
    )
    parser.add_argument(
        "--train-subpath",
        action="append",
        dest="train_subpaths",
        help="Candidate sub‑folders with clean images; can be given multiple times",
    )
    parser.add_argument(
        "--keep-names",
        action="store_true",
        help="Preserve original filenames",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = {}

    if args.config:
        cfg = load_config(args.config)

    def pick(key: str, *aliases):
        for k in (key, *aliases):
            val = getattr(args, k, None)
            if val is not None:
                return val
        return cfg.get(key)

    source = pick("source")
    target = pick("target")
    if source is None or target is None:
        sys.exit("Both 'source' and 'target' must be specified (CLI or config).")

    num_samples = pick("num_samples") or 1
    rename_images = bool(pick("keep_names"))

    sample_images(
        source_root=Path(source),
        target_root=Path(target),
        num_samples=int(num_samples),
        rename_images=rename_images,
    )


if __name__ == "__main__":
    main()