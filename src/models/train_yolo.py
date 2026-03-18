from ultralytics import YOLO
from pathlib import Path

# =========================================
# 🔧 설정
# =========================================
MODEL_NAME = "yolov8n.pt"
DATA_YAML = "yolo_dataset/data.yaml"

EPOCHS = 50
IMG_SIZE = 640
BATCH = 8
DEVICE = "cpu"  # GPU면 0

RUN_NAME = "exp1"


# =========================================
# 🚀 학습
# =========================================
def main():
    print("🚀 YOLOv8 Training Start")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        val=False,          # 🔥 validation 끔
        name=RUN_NAME,
        exist_ok=True,
        verbose=True
    )

    # =========================================
    # 📁 결과 경로 출력
    # =========================================
    save_dir = Path(results.save_dir)

    print("\n✅ Training Finished")
    print(f"📁 save_dir: {save_dir}")

    # weights
    last_pt = save_dir / "weights" / "last.pt"

    if last_pt.exists():
        print(f"✅ last.pt: {last_pt}")
    else:
        print("❌ last.pt 없음")

    # csv
    csv_path = save_dir / "results.csv"

    if csv_path.exists():
        print(f"✅ results.csv: {csv_path}")
    else:
        print("❌ results.csv 없음")


# =========================================
# 실행
# =========================================
if __name__ == "__main__":
    main()