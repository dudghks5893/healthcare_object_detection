from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch
import wandb
import random
import numpy as np

# =========================================
# Seed 고정
# =========================================
SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 재현성 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# =========================================
# 1. 기본 설정
# =========================================
# 사전학습 모델
MODEL_NAME = "yolov8m.pt"

# 기존 로컬 프로젝트 경로 유지
BASE_DIR = Path(r"C:\Users\jgi01\OneDrive\바탕 화면\Codit_ML\healthcare_object_detection")

# YOLO 데이터셋 경로
DATASET_DIR = BASE_DIR / "yolo_dataset_all_train"

# 이미지 / 라벨 경로
TRAIN_IMG_DIR = DATASET_DIR / "images"
TRAIN_LBL_DIR = DATASET_DIR / "labels"

# data.yaml 경로
DATA_YAML = DATASET_DIR / "data.yaml"

# =========================================
# 2. 학습 하이퍼파라미터
# =========================================
EPOCHS = 300
IMG_SIZE = 960
BATCH = 16

# GPU 사용 가능하면 GPU 사용
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# 실험 이름
RUN_NAME = "yolov8m_remove_aug_300epoch"

# =========================================
# 3. Optimizer / LR / Scheduler 설정
# =========================================
OPTIMIZER = "SGD"
LR0 = 0.01
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3
COS_LR = True

# =========================================
# 4. wandb 초기화
# =========================================
wandb.init(
    entity="sprint_AI_9th_Team_3",
    project="pill-ssd",
    name=RUN_NAME,
    config={
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "device": DEVICE,
        "run_name": RUN_NAME,
        "data_yaml": str(DATA_YAML),

        # augmentation 설정 기록
        "mosaic": 0.0,
        "erasing": 0.0,
        "scale": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "fliplr": 0.5,
        "flipud": 0.5,
        "dropout": 0.3,

        # optimizer / lr 설정 기록
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "cos_lr": COS_LR,
        "seed": SEED,
    }
)

# =========================================
# 5. 학습 함수
# =========================================
def main():
    print("🚀 Training Start")

    # 현재 실행 환경 출력
    print(f"🔧 device: {'GPU' if DEVICE != 'cpu' else 'CPU'}")

    # 주요 경로 확인
    print(f"📂 DATA_YAML: {DATA_YAML.resolve()}")
    print(f"🖼️ TRAIN_IMG_DIR: {TRAIN_IMG_DIR.resolve()}")
    print(f"🏷️ TRAIN_LBL_DIR: {TRAIN_LBL_DIR.resolve()}")

    # 필수 파일/폴더 존재 확인
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    if not TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"train images dir not found: {TRAIN_IMG_DIR}")

    if not TRAIN_LBL_DIR.exists():
        raise FileNotFoundError(f"train labels dir not found: {TRAIN_LBL_DIR}")

    # data.yaml 내용 출력
    print("\n===== data.yaml contents =====")
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        print(f.read())
    print("===== end =====\n")

    # 모델 로드
    model = YOLO(MODEL_NAME)

    # 학습 실행
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,

        # augmentation 설정
        mosaic=0.0,
        erasing=0.0,
        scale=0.0,
        augment=False,
        dropout=0.3,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.5,
        flipud=0.5,

        # 검증 없이 train만 수행
        val=False,

        # 저장 폴더 이름
        name=RUN_NAME,
        exist_ok=True,

        # 재현성
        seed=SEED,

        # 로그 출력
        verbose=True,

        # optimizer / lr / scheduler
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        cos_lr=COS_LR,
    )

    # 저장 경로 확인
    save_dir = Path(results.save_dir)
    print(f"\n📁 save_dir: {save_dir}")

    # results.csv를 읽어 wandb에 기록
    csv_path = save_dir / "results.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print("📊 wandb logging...")

        for _, row in df.iterrows():
            wandb.log({
                "epoch": int(row["epoch"]),
                "box_loss": float(row.get("train/box_loss", 0)),
                "cls_loss": float(row.get("train/cls_loss", 0)),
                "dfl_loss": float(row.get("train/dfl_loss", 0)),
                "lr_pg0": float(row.get("lr/pg0", 0)),
                "lr_pg1": float(row.get("lr/pg1", 0)),
                "lr_pg2": float(row.get("lr/pg2", 0)),
            })

        print("✅ wandb logging 완료")
    else:
        print("❌ results.csv 없음")

    # 마지막 / 최고 성능 weight 파일 확인
    last_pt = save_dir / "weights" / "last.pt"
    best_pt = save_dir / "weights" / "best.pt"

    if last_pt.exists():
        print(f"✅ last.pt: {last_pt}")
    else:
        print("❌ last.pt 없음")

    if best_pt.exists():
        print(f"✅ best.pt: {best_pt}")
    else:
        print("❌ best.pt 없음")

    # wandb 종료
    wandb.finish()

# =========================================
# 6. 실행
# =========================================
if __name__ == "__main__":
    main()