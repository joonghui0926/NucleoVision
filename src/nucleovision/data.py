from __future__ import annotations

import csv
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import yaml
from PIL import Image
from tqdm import tqdm

from . import settings


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    source_image: Path
    mask_paths: tuple[Path, ...]
    width: int
    height: int
    actual_count: int
    mean_mask_area: float
    mask_density: float


def ensure_dirs() -> None:
    for path in [
        settings.RAW_ROOT,
        settings.PROCESSED_ROOT,
        settings.YOLO_ROOT,
        settings.OUTPUT_ROOT,
        settings.MODEL_OUTPUT,
        settings.TABLE_OUTPUT,
        settings.PLOT_OUTPUT,
        settings.PREDICTION_OUTPUT,
        settings.REPORT_OUTPUT,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def download_bbbc038(force: bool = False) -> Path:
    ensure_dirs()
    if settings.STAGE1_ZIP.exists() and not force:
        return settings.STAGE1_ZIP

    response = requests.get(settings.BBBC038_URL, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with settings.STAGE1_ZIP.open("wb") as file_obj:
        progress = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading BBBC038")
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_obj.write(chunk)
                progress.update(len(chunk))
        progress.close()

    return settings.STAGE1_ZIP


def has_extracted_samples(path: Path) -> bool:
    if not path.exists():
        return False
    return any(
        sample_dir.is_dir() and (sample_dir / "images").exists() and (sample_dir / "masks").exists()
        for sample_dir in path.iterdir()
    )


def extract_bbbc038(force: bool = False) -> Path:
    if has_extracted_samples(settings.STAGE1_DIR) and not force:
        return settings.STAGE1_DIR

    if force and settings.RAW_ROOT.exists():
        for child in settings.RAW_ROOT.iterdir():
            if child.resolve() == settings.STAGE1_ZIP.resolve():
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    with zipfile.ZipFile(settings.STAGE1_ZIP) as archive:
        archive.extractall(settings.RAW_ROOT)

    return settings.STAGE1_DIR


def read_mask_stats(mask_paths: tuple[Path, ...]) -> tuple[int, float]:
    areas = []
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L"))
        areas.append(int((mask > 0).sum()))
    if not areas:
        return 0, 0.0
    return len(areas), float(np.mean(areas))


def load_records() -> list[ImageRecord]:
    records = []
    for sample_dir in sorted(path for path in settings.STAGE1_DIR.iterdir() if path.is_dir()):
        image_paths = sorted((sample_dir / "images").glob("*.png"))
        mask_paths = tuple(sorted((sample_dir / "masks").glob("*.png")))
        if not image_paths or not mask_paths:
            continue

        with Image.open(image_paths[0]) as image:
            width, height = image.size

        actual_count, mean_mask_area = read_mask_stats(mask_paths)
        records.append(
            ImageRecord(
                image_id=sample_dir.name,
                source_image=image_paths[0],
                mask_paths=mask_paths,
                width=width,
                height=height,
                actual_count=actual_count,
                mean_mask_area=mean_mask_area,
                mask_density=actual_count / float(width * height),
            )
        )
    return records


def select_records(records: list[ImageRecord], seed: int, total: int) -> list[ImageRecord]:
    if len(records) < total:
        raise ValueError(f"BBBC038 contains {len(records)} usable images, but {total} are required.")
    rng = random.Random(seed)
    selected = records[:]
    rng.shuffle(selected)
    return selected[:total]


def split_records(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    split_map = {}
    cursor = 0
    for split_name, split_size in settings.SPLIT_SIZES.items():
        split_map[split_name] = records[cursor : cursor + split_size]
        cursor += split_size
    return split_map


def mask_to_box(mask_path: Path, width: int, height: int) -> tuple[float, float, float, float, int] | None:
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    box_width = xmax - xmin + 1
    box_height = ymax - ymin + 1
    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0
    area = box_width * box_height

    return (
        x_center / width,
        y_center / height,
        box_width / width,
        box_height / height,
        area,
    )


def write_label_file(record: ImageRecord, label_path: Path) -> float:
    box_areas = []
    lines = []
    for mask_path in record.mask_paths:
        box = mask_to_box(mask_path, record.width, record.height)
        if box is None:
            continue
        x_center, y_center, box_width, box_height, area = box
        box_areas.append(area)
        lines.append(f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}")

    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return float(np.mean(box_areas)) if box_areas else 0.0


def copy_image_as_rgb(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        image.convert("RGB").save(target)


def prepare_yolo_dataset(force: bool = False, seed: int = settings.RANDOM_SEED) -> None:
    ensure_dirs()
    total = sum(settings.SPLIT_SIZES.values())

    if settings.SPLITS_CSV.exists() and settings.DATA_YAML.exists() and not force:
        return

    if settings.YOLO_ROOT.exists():
        shutil.rmtree(settings.YOLO_ROOT)
    settings.YOLO_ROOT.mkdir(parents=True, exist_ok=True)

    records = select_records(load_records(), seed=seed, total=total)
    split_map = split_records(records)
    metadata_rows = []

    for split_name, split_records_ in split_map.items():
        image_dir = settings.YOLO_ROOT / "images" / split_name
        label_dir = settings.YOLO_ROOT / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for index, record in enumerate(split_records_, start=1):
            output_name = f"{split_name}_{index:03d}_{record.image_id}.png"
            image_path = image_dir / output_name
            label_path = label_dir / output_name.replace(".png", ".txt")
            copy_image_as_rgb(record.source_image, image_path)
            mean_box_area = write_label_file(record, label_path)
            metadata_rows.append(
                {
                    "image_id": record.image_id,
                    "split": split_name,
                    "file_name": output_name,
                    "width": record.width,
                    "height": record.height,
                    "actual_count": record.actual_count,
                    "mean_mask_area": round(record.mean_mask_area, 4),
                    "mean_box_area": round(mean_box_area, 4),
                    "mask_density": record.mask_density,
                    "image_path": image_path.relative_to(settings.PROJECT_ROOT).as_posix(),
                    "label_path": label_path.relative_to(settings.PROJECT_ROOT).as_posix(),
                }
            )

    write_metadata(metadata_rows)
    write_data_yaml()


def write_metadata(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image_id",
        "split",
        "file_name",
        "width",
        "height",
        "actual_count",
        "mean_mask_area",
        "mean_box_area",
        "mask_density",
        "image_path",
        "label_path",
    ]
    with settings.SPLITS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    shutil.copyfile(settings.SPLITS_CSV, settings.IMAGE_METADATA_CSV)


def write_data_yaml() -> None:
    data = {
        "path": settings.YOLO_ROOT.relative_to(settings.PROJECT_ROOT).as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "nucleus"},
    }
    settings.DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def prepare_all(force_download: bool = False, force_prepare: bool = False, seed: int = settings.RANDOM_SEED) -> None:
    download_bbbc038(force=force_download)
    extract_bbbc038(force=force_download)
    prepare_yolo_dataset(force=force_prepare, seed=seed)
