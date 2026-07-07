from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from . import settings


def choose_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def train_model(
    epochs: int,
    imgsz: int,
    batch: int,
    seed: int,
    device: str | None = None,
    workers: int = 0,
    force: bool = False,
) -> Path:
    settings.MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    if settings.BEST_MODEL.exists() and not force:
        return settings.BEST_MODEL

    selected_device = device or choose_device()
    model = YOLO("yolo11n.pt")
    result = model.train(
        data=str(settings.DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        deterministic=True,
        device=selected_device,
        workers=workers,
        project=str(settings.RUNS_ROOT),
        name="nuclei_yolo11n",
        exist_ok=True,
        single_cls=True,
        plots=True,
        verbose=True,
    )

    save_dir = Path(result.save_dir)
    best_source = save_dir / "weights" / "best.pt"
    last_source = save_dir / "weights" / "last.pt"
    shutil.copyfile(best_source, settings.BEST_MODEL)
    shutil.copyfile(last_source, settings.LAST_MODEL)

    for name in ["results.csv", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
        source = save_dir / name
        if source.exists():
            shutil.copyfile(source, settings.MODEL_OUTPUT / name)

    config = {
        "model": "yolo11n.pt",
        "data": settings.DATA_YAML.relative_to(settings.PROJECT_ROOT).as_posix(),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": seed,
        "device": selected_device,
        "workers": workers,
        "single_cls": True,
    }
    (settings.MODEL_OUTPUT / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return settings.BEST_MODEL
