# NucleoVision

YOLO11n을 이용해 BBBC038 세포핵 현미경 이미지를 탐지하고, 세포핵 개수, 크기, 밀도, 탐지 오차율을 정량화하는 탐구 프로젝트입니다. 이 저장소는 모든 실험을 그대로 실행할 수 있도록 구성했습니다.

## 실험 개요

- 데이터셋: BBBC038, Kaggle 2018 Data Science Bowl nuclei dataset
- 모델: Ultralytics YOLO11n
- 데이터 구성: train 100장, validation 30장, test 30장
- 라벨 변환: 세포핵 mask 1개를 YOLO bounding box 1개로 변환
- 분석 항목: 실제 세포핵 수, YOLO 탐지 수, 오차율, 평균 box 크기, 세포핵 밀도
- 밀도 분석: test 30장을 실제 세포핵 수 기준으로 low, medium, high 각 10장씩 배정

## 현재 생성된 결과

이번 실행은 CPU 환경에서 20 epoch로 진행했습니다.

- 테스트 이미지 수: 30장
- 실제 세포핵 총수: 1162개
- YOLO 탐지 세포핵 총수: 1166개
- 평균 탐지 오차율: 7.80%
- 중앙값 탐지 오차율: 5.28%
- 평균 confidence: 0.790
- test mAP50: 0.828
- test mAP50-95: 0.557
- test precision: 0.888
- test recall: 0.818

## 폴더 구조

```text
.
├── data/
│   └── processed/
│       ├── splits.csv
│       ├── image_metadata.csv
│       └── yolo/
│           ├── nuclei.yaml
│           ├── images/
│           └── labels/
├── outputs/
│   ├── model/
│   ├── plots/
│   ├── predictions/
│   ├── report/
│   ├── tables/
│   └── validation/
├── src/
│   └── nucleovision/
├── run_pipeline.py
├── requirements.txt
```

`data/raw`는 BBBC038 원본 ZIP과 압축 해제 파일이 들어가는 위치이며 Git에는 포함하지 않습니다. 전체 재실행 시 자동으로 다시 다운로드됩니다.

## 산출물 위치

- 실험 요약 보고서: `outputs/report/experiment_summary.md`
- 이미지별 결과표: `outputs/tables/test_metrics.csv`
- 밀도 그룹별 결과표: `outputs/tables/density_group_metrics.csv`
- 전체 요약 지표: `outputs/tables/summary_metrics.json`
- 객체 탐지 검증 지표: `outputs/tables/detection_validation_metrics.json`
- 실제 개수와 예측 개수 비교 그래프: `outputs/plots/actual_vs_predicted_counts.png`
- 이미지별 오차율 그래프: `outputs/plots/detection_error_by_image.png`
- 밀도 그룹별 평균 오차율 그래프: `outputs/plots/density_group_error.png`
- 예측 이미지 4장: `outputs/predictions/examples`
- 테스트 이미지 전체 예측 결과: `outputs/predictions/test_images`
- 학습된 best 모델: `outputs/model/yolo11n_nuclei_best.pt`
- 학습된 last 모델: `outputs/model/yolo11n_nuclei_last.pt`

## 재현 방법

Windows PowerShell 기준입니다. Python 3.12 사용을 권장합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

전체 실험을 처음부터 다시 실행하려면 아래 명령을 사용합니다.

```powershell
python run_pipeline.py --epochs 20 --imgsz 512 --batch 8 --seed 42 --conf 0.25 --iou 0.5 --force-train
```

원본 데이터를 다시 받고 전처리까지 새로 하려면 아래처럼 실행합니다.

```powershell
python run_pipeline.py --epochs 20 --imgsz 512 --batch 8 --seed 42 --conf 0.25 --iou 0.5 --force-download --force-prepare --force-train
```

CPU에서는 약 8분 정도 학습 시간이 걸렸습니다. GPU가 있는 환경에서는 더 빠르게 끝납니다.

## 파이프라인 동작

1. `data/raw/stage1_train.zip`이 없으면 BBBC038 원본 데이터를 다운로드합니다.
2. ZIP을 풀고 이미지와 mask 폴더를 읽습니다.
3. seed 42로 160장을 선정해 train 100장, validation 30장, test 30장으로 나눕니다.
4. mask 파일별 픽셀 범위를 계산해 YOLO bounding box 라벨로 저장합니다.
5. YOLO11n pretrained weight를 불러와 세포핵 1개 class로 학습합니다.
6. test 30장에 대해 예측 box를 만들고 실제 mask 개수와 비교합니다.
7. 표, 그래프, 예측 이미지, 요약 보고서를 `outputs`에 저장합니다.

## 해석 기준

이 프로젝트의 결과는 현미경 이미지에서 관찰 가능한 형태 정보를 정량화한 것입니다. 세포핵 개수와 밀도는 세포 증식이나 이미지 내 밀집 정도를 해석하는 보조 지표가 될 수 있지만, 이미지 분석만으로 암세포 대사나 산화 스트레스를 직접 측정했다고 볼 수는 없습니다.
