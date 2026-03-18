from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch
import wandb

# =========================================
# 설정
# =========================================
MODEL_NAME = "yolov8n.pt"
DATA_YAML = "yolo_dataset_all_train/data.yaml"

EPOCHS = 20
IMG_SIZE = 640
BATCH = 8
DEVICE = 0 if torch.cuda.is_available() else "cpu"

RUN_NAME = "yolo_test_exp"

# =========================================
# wandb 시작
# =========================================
wandb.init(
    entity="sprint_AI_9th_Team_3",
    project="pill-ssd",   # 🔥 프로젝트 이름
    name=RUN_NAME,
    config={
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": BATCH,
    }
)

# =========================================
# 학습
# =========================================
def main():
    print("🚀 Training Start")
    print(f"🔧 device: {'GPU' if DEVICE != 'cpu' else 'CPU'}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
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

    # =========================================
    # results.csv → wandb 로그
    # =========================================
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

    # =========================================
    # weight 확인
    # =========================================
    last_pt = save_dir / "weights" / "last.pt"

    if last_pt.exists():
        print(f"✅ last.pt: {last_pt}")
    else:
        print("❌ last.pt 없음")

    wandb.finish()


# =========================================
# 실행
# =========================================
if __name__ == "__main__":
    main()