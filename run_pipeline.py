from __future__ import annotations

import argparse

from src.nucleovision import settings
from src.nucleovision.data import prepare_all
from src.nucleovision.evaluate import evaluate_model
from src.nucleovision.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NucleoVision YOLO11n experiment.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_all(force_download=args.force_download, force_prepare=args.force_prepare, seed=args.seed)
    model_path = train_model(
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        force=args.force_train,
    )
    evaluate_model(model_path=model_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou, batch=args.batch)


if __name__ == "__main__":
    main()
