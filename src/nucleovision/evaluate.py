from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ultralytics import YOLO

from . import settings


matplotlib.use("Agg")


def project_path_text(path: Path) -> str:
    absolute_path = path if path.is_absolute() else settings.PROJECT_ROOT / path
    return absolute_path.relative_to(settings.PROJECT_ROOT).as_posix()


def load_test_metadata() -> pd.DataFrame:
    metadata = pd.read_csv(settings.SPLITS_CSV)
    test_data = metadata[metadata["split"] == "test"].copy()
    test_data["image_abs_path"] = test_data["image_path"].apply(lambda p: str(settings.PROJECT_ROOT / p))
    return test_data.reset_index(drop=True)


def assign_density_groups(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("actual_count").copy()
    group_names = ["low", "medium", "high"]
    groups = []
    for index in range(len(ordered)):
        groups.append(group_names[min(index // 10, 2)])
    ordered["density_group"] = groups
    return ordered.sort_index()


def save_prediction_image(result, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    plotted = result.plot(line_width=1, font_size=8)
    cv2.imwrite(str(target_path), plotted)


def evaluate_model(model_path: Path, imgsz: int, conf: float, iou: float, batch: int) -> None:
    settings.TABLE_OUTPUT.mkdir(parents=True, exist_ok=True)
    settings.PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)
    settings.PREDICTION_OUTPUT.mkdir(parents=True, exist_ok=True)
    settings.REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    model.model.names = {0: "nucleus"}
    test_data = assign_density_groups(load_test_metadata())

    rows = []
    example_paths = []
    prediction_dir = settings.PREDICTION_OUTPUT / "test_images"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for row_index, row in test_data.iterrows():
        result = model.predict(
            source=row["image_abs_path"],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False,
        )[0]
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.empty((0,))
        pred_count = int(len(boxes))
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if pred_count else np.array([])
        image_area = float(row["width"] * row["height"])
        actual_count = int(row["actual_count"])
        error_rate = abs(actual_count - pred_count) / actual_count * 100.0 if actual_count else 0.0
        mean_confidence = float(confidences.mean()) if pred_count else 0.0
        mean_pred_box_area = float(box_areas.mean()) if pred_count else 0.0

        prediction_path = prediction_dir / row["file_name"].replace(".png", "_prediction.jpg")
        save_prediction_image(result, prediction_path)
        if len(example_paths) < 4:
            example_paths.append(prediction_path)

        rows.append(
            {
                "file_name": row["file_name"],
                "image_id": row["image_id"],
                "density_group": row["density_group"],
                "width": int(row["width"]),
                "height": int(row["height"]),
                "actual_count": actual_count,
                "predicted_count": pred_count,
                "count_difference": pred_count - actual_count,
                "error_rate_percent": round(error_rate, 4),
                "mean_mask_area": round(float(row["mean_mask_area"]), 4),
                "mean_ground_truth_box_area": round(float(row["mean_box_area"]), 4),
                "mean_predicted_box_area": round(mean_pred_box_area, 4),
                "actual_density": row["mask_density"],
                "predicted_density": pred_count / image_area,
                "mean_confidence": round(mean_confidence, 4),
                "prediction_image": project_path_text(prediction_path),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics_path = settings.TABLE_OUTPUT / "test_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8")

    group_metrics = (
        metrics.groupby("density_group", sort=False)
        .agg(
            images=("file_name", "count"),
            mean_actual_count=("actual_count", "mean"),
            mean_predicted_count=("predicted_count", "mean"),
            mean_error_rate_percent=("error_rate_percent", "mean"),
            mean_predicted_box_area=("mean_predicted_box_area", "mean"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
    )
    group_order = ["low", "medium", "high"]
    group_metrics["density_group"] = pd.Categorical(
        group_metrics["density_group"],
        categories=group_order,
        ordered=True,
    )
    group_metrics = group_metrics.sort_values("density_group")
    group_metrics["density_group"] = group_metrics["density_group"].astype(str)
    group_path = settings.TABLE_OUTPUT / "density_group_metrics.csv"
    group_metrics.to_csv(group_path, index=False, encoding="utf-8")

    summary = {
        "test_images": int(len(metrics)),
        "total_actual_nuclei": int(metrics["actual_count"].sum()),
        "total_predicted_nuclei": int(metrics["predicted_count"].sum()),
        "mean_error_rate_percent": float(metrics["error_rate_percent"].mean()),
        "median_error_rate_percent": float(metrics["error_rate_percent"].median()),
        "mean_confidence": float(metrics["mean_confidence"].mean()),
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "image_size": imgsz,
        "model_path": project_path_text(model_path),
    }
    (settings.TABLE_OUTPUT / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_example_predictions(example_paths)
    save_plots(metrics, group_metrics)
    save_validation_outputs(model, imgsz=imgsz, conf=conf, iou=iou, batch=batch)
    write_report(metrics, group_metrics, summary)


def save_example_predictions(example_paths: list[Path]) -> None:
    example_dir = settings.PREDICTION_OUTPUT / "examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(example_paths, start=1):
        shutil.copyfile(source, example_dir / f"example_{index}.jpg")


def save_plots(metrics: pd.DataFrame, group_metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("file_name").reset_index(drop=True)
    x = np.arange(len(ordered))

    plt.figure(figsize=(14, 6))
    plt.bar(x - 0.2, ordered["actual_count"], width=0.4, label="Actual")
    plt.bar(x + 0.2, ordered["predicted_count"], width=0.4, label="YOLO")
    plt.xticks(x, [f"{i + 1}" for i in range(len(ordered))], fontsize=8)
    plt.xlabel("Test image")
    plt.ylabel("Nucleus count")
    plt.title("Actual and YOLO detected nuclei")
    plt.legend()
    plt.tight_layout()
    plt.savefig(settings.PLOT_OUTPUT / "actual_vs_predicted_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(x, ordered["error_rate_percent"], marker="o", linewidth=1.8)
    plt.xticks(x, [f"{i + 1}" for i in range(len(ordered))], fontsize=8)
    plt.xlabel("Test image")
    plt.ylabel("Error rate (%)")
    plt.title("Detection error rate by test image")
    plt.tight_layout()
    plt.savefig(settings.PLOT_OUTPUT / "detection_error_by_image.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(group_metrics["density_group"], group_metrics["mean_error_rate_percent"])
    plt.xlabel("Density group")
    plt.ylabel("Mean error rate (%)")
    plt.title("Mean detection error by density group")
    plt.tight_layout()
    plt.savefig(settings.PLOT_OUTPUT / "density_group_error.png", dpi=200)
    plt.close()


def save_validation_outputs(model: YOLO, imgsz: int, conf: float, iou: float, batch: int) -> None:
    result = model.val(
        data=str(settings.DATA_YAML),
        split="test",
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        project=str(settings.OUTPUT_ROOT / "validation"),
        name="test_set",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    stats = {
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
    }
    (settings.TABLE_OUTPUT / "detection_validation_metrics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_report(metrics: pd.DataFrame, group_metrics: pd.DataFrame, summary: dict[str, object]) -> None:
    best_examples = metrics.sort_values("error_rate_percent").head(3)
    difficult_examples = metrics.sort_values("error_rate_percent", ascending=False).head(3)

    lines = [
        "# YOLO 기반 세포 이미지 정량화 결과",
        "",
        "## 실행 요약",
        f"- 테스트 이미지 수: {summary['test_images']}",
        f"- 실제 세포핵 총수: {summary['total_actual_nuclei']}",
        f"- YOLO 탐지 세포핵 총수: {summary['total_predicted_nuclei']}",
        f"- 평균 탐지 오차율: {summary['mean_error_rate_percent']:.2f}%",
        f"- 중앙값 탐지 오차율: {summary['median_error_rate_percent']:.2f}%",
        f"- 평균 confidence: {summary['mean_confidence']:.3f}",
        "",
        "## 실험 A. YOLO11n 세포핵 탐지 가능성",
        "테스트 이미지 전체에 대해 예측 이미지를 저장했다. 예시 이미지는 `outputs/predictions/examples`에서 확인할 수 있다.",
        "",
        "## 실험 B. 세포핵 개수 정량화",
        "이미지별 실제 세포핵 수, YOLO 탐지 수, 오차율은 `outputs/tables/test_metrics.csv`에 저장했다.",
        "실제 개수와 예측 개수 비교 그래프는 `outputs/plots/actual_vs_predicted_counts.png`에 저장했다.",
        "",
        "## 실험 C. 밀집도에 따른 탐지 오차 분석",
        "테스트 이미지 30장을 실제 세포핵 수 기준으로 낮은 밀도, 중간 밀도, 높은 밀도 그룹에 10장씩 배정했다.",
    ]

    for _, row in group_metrics.iterrows():
        lines.append(
            f"- {row['density_group']}: 평균 실제 개수 {row['mean_actual_count']:.1f}, "
            f"평균 탐지 개수 {row['mean_predicted_count']:.1f}, 평균 오차율 {row['mean_error_rate_percent']:.2f}%"
        )

    lines.extend(
        [
            "",
            "## 오차가 낮은 예시",
        ]
    )
    for _, row in best_examples.iterrows():
        lines.append(
            f"- {row['file_name']}: 실제 {row['actual_count']}, 예측 {row['predicted_count']}, 오차율 {row['error_rate_percent']:.2f}%"
        )

    lines.extend(
        [
            "",
            "## 오차가 높은 예시",
        ]
    )
    for _, row in difficult_examples.iterrows():
        lines.append(
            f"- {row['file_name']}: 실제 {row['actual_count']}, 예측 {row['predicted_count']}, 오차율 {row['error_rate_percent']:.2f}%"
        )

    lines.extend(
        [
            "",
            "## 해석",
            "이 결과는 현미경 이미지의 세포핵을 객체 탐지 문제로 바꾸고, 개수와 크기, 밀도 같은 형태 지표를 표와 그래프로 정량화한 것이다.",
            "세포핵이 많거나 서로 겹치는 이미지에서는 누락이나 중복 탐지가 늘어날 수 있으므로, 결과 해석은 이미지 기반 형태 분석의 보조 지표로 제한해야 한다.",
            "이 실험만으로 암세포 대사나 산화 스트레스를 직접 측정했다고 볼 수는 없다.",
        ]
    )

    (settings.REPORT_OUTPUT / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
