from ultralytics import YOLO
from pathlib import Path
from src.utils import get_device
from src.utils import set_seed

"""
    실행 순서: 4
    
    실행 방법:
    터미널에 python -m src.engine.train_stage1_detector 입력

    [역할]
    YOLO 기반 객체 탐지 모델(Stage1 Detector)을 학습하는 단계

    [하는 일]
    - YOLO 모델 로드 및 학습 수행
    - bbox 검출 성능 학습

    [결과]
    best.pt / last.pt (detector weight)
"""

DATA_YAML = Path("data/processed/yolo_stage1/data.yaml")
SEED = 42

def main():
    set_seed(SEED)
    device = get_device()

    # YOLO용 device 변환
    if device.type == "cuda":
        yolo_device = "0"
    elif device.type == "mps":
        yolo_device = "mps"
    else:
        yolo_device = "cpu"

    # 소형 모델로 시작
    model = YOLO("yolo11n.pt")  # 설치 버전에 따라 yolo11n.pt 또는 최신 detect nano 사용 가능

    results = model.train(
        data=str(DATA_YAML),
        epochs=20,
        imgsz=960,
        batch=8,
        device=yolo_device,          # GPU면 0, CPU면 "cpu"
        seed=SEED,
        workers=4,
        project="outputs/stage1_detector",
        name="baseline",
        pretrained=True,
        patience=15,
        save=True,
        verbose=True
    )

    print(results)

if __name__ == "__main__":
    main()