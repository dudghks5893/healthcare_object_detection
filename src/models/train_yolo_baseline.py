from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch
import wandb

# =========================================
# 설정
# =========================================
MODEL_NAME = "yolov8m.pt"

BASE_DIR = Path(r"C:\Users\jgi01\OneDrive\바탕 화면\Codit_ML\healthcare_object_detection")
DATASET_DIR = BASE_DIR / "mini_project_dataset.v1-train_dataset_version2.yolov8"

TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
DATA_YAML = DATASET_DIR / "data.yaml"

EPOCHS = 20
IMG_SIZE = 640
BATCH = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

RUN_NAME = "yolov8m_2nd_data"

# =========================================
# wandb 시작
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
        "data_yaml": str(DATA_YAML),
        "train_images": str(TRAIN_IMG_DIR),
        "train_labels": str(TRAIN_LBL_DIR),
    }
)

# =========================================
# 학습
# =========================================
def main():
    print("🚀 Training Start")
    print(f"🔧 device: {'GPU' if DEVICE != 'cpu' else 'CPU'}")
    print(f"📂 DATA_YAML: {DATA_YAML}")
    print(f"🖼️ TRAIN_IMG_DIR: {TRAIN_IMG_DIR}")
    print(f"🏷️ TRAIN_LBL_DIR: {TRAIN_LBL_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")
    if not TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"train images dir not found: {TRAIN_IMG_DIR}")
    if not TRAIN_LBL_DIR.exists():
        raise FileNotFoundError(f"train labels dir not found: {TRAIN_LBL_DIR}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        val=False,
        name=RUN_NAME,
        exist_ok=True,
        verbose=True
    )

    save_dir = Path(results.save_dir)
    print(f"\n📁 save_dir: {save_dir}")

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
            })

        print("✅ wandb logging 완료")
    else:
        print("❌ results.csv 없음")

    last_pt = save_dir / "weights" / "last.pt"

    if last_pt.exists():
        print(f"✅ last.pt: {last_pt}")
    else:
        print("❌ last.pt 없음")

    wandb.finish()

if __name__ == "__main__":
    main()