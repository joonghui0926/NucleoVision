from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
YOLO_ROOT = PROCESSED_ROOT / "yolo"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RUNS_ROOT = PROJECT_ROOT / "runs"

BBBC038_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"
STAGE1_ZIP = RAW_ROOT / "stage1_train.zip"
STAGE1_DIR = RAW_ROOT

SPLIT_SIZES = {"train": 100, "val": 30, "test": 30}
RANDOM_SEED = 42

DATA_YAML = YOLO_ROOT / "nuclei.yaml"
SPLITS_CSV = PROCESSED_ROOT / "splits.csv"
IMAGE_METADATA_CSV = PROCESSED_ROOT / "image_metadata.csv"

MODEL_OUTPUT = OUTPUT_ROOT / "model"
TABLE_OUTPUT = OUTPUT_ROOT / "tables"
PLOT_OUTPUT = OUTPUT_ROOT / "plots"
PREDICTION_OUTPUT = OUTPUT_ROOT / "predictions"
REPORT_OUTPUT = OUTPUT_ROOT / "report"

BEST_MODEL = MODEL_OUTPUT / "yolo11n_nuclei_best.pt"
LAST_MODEL = MODEL_OUTPUT / "yolo11n_nuclei_last.pt"
