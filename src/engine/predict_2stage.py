from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO

from src.utils import get_device

"""
    실행 순서: 8

    실행 방법:
    터미널에 python -m src.engine.predict_2stage 입력

    [역할]
    Stage1 Detector + Stage2 Classifier를 결합하여 최종 예측 결과를 생성하는 단계

    [하는 일]
    1. detector로 bbox 검출
    2. bbox 영역 crop
    3. classifier로 클래스 예측
    4. 제출 형식 CSV 생성

    [결과]
    predict_2stage.csv (annotation_id, image_id, category_id, bbox, score)
"""

# ===== 경로 설정 =====
TEST_IMG_DIR = Path("data/raw/test_images")
DETECTOR_WEIGHT = Path("runs/detect/outputs/stage1_detector/baseline/weights/best.pt")
CLASSIFIER_WEIGHT = Path("outputs/stage2_classifier/resnet18_fulltrain/best.pt")

SAVE_DIR = Path("outputs/submission")
SAVE_CSV = SAVE_DIR / "predict_2stage.csv"

# ===== detector 설정 =====
DET_IMGSZ = 960
DET_CONF = 0.25
DET_IOU = 0.50

# ===== classifier 설정 =====
CLS_IMGSZ = 224
CROP_MARGIN_RATIO = 0.10


def build_classifier(num_classes):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(weight_path, device):
    ckpt = torch.load(weight_path, map_location=device)

    num_classes = ckpt["num_classes"]
    idx_to_class = ckpt["idx_to_class"]

    # key가 문자열일 수 있으니 int로 변환
    idx_to_class = {int(k): str(v) for k, v in idx_to_class.items()}

    model = build_classifier(num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, idx_to_class


def build_classifier_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def classify_crop(crop_img, classifier, transform, device, idx_to_class):
    x = transform(crop_img).unsqueeze(0).to(device)
    logits = classifier(x)
    probs = F.softmax(logits, dim=1)

    conf, pred_idx = probs.max(dim=1)
    pred_idx = int(pred_idx.item())
    conf = float(conf.item())

    pred_class_id = int(idx_to_class[pred_idx])
    return pred_class_id, conf


def apply_margin_and_clip(x1, y1, x2, y2, img_w, img_h, margin_ratio=0.10):
    w = x2 - x1
    h = y2 - y1

    mx = w * margin_ratio
    my = h * margin_ratio

    nx1 = max(0, int(x1 - mx))
    ny1 = max(0, int(y1 - my))
    nx2 = min(img_w, int(x2 + mx))
    ny2 = min(img_h, int(y2 + my))

    return nx1, ny1, nx2, ny2


def extract_image_id(file_name: str) -> int:
    # 파일명이 000123.png 형태라고 가정
    # stem만 int로 변환
    return int(Path(file_name).stem)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    if device.type == "cuda":
        yolo_device = "0"
    elif device.type == "mps":
        yolo_device = "mps"
    else:
        yolo_device = "cpu"

    detector = YOLO(str(DETECTOR_WEIGHT))
    classifier, idx_to_class = load_classifier(CLASSIFIER_WEIGHT, device)
    cls_transform = build_classifier_transform(CLS_IMGSZ)

    image_paths = sorted(TEST_IMG_DIR.glob("*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    print(f"test 이미지 수: {len(image_paths)}")

    rows = []
    annotation_id = 1

    for img_path in image_paths:
        file_name = img_path.name
        image_id = extract_image_id(file_name)

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        results = detector.predict(
            source=str(img_path),
            imgsz=DET_IMGSZ,
            conf=DET_CONF,
            iou=DET_IOU,
            device=yolo_device,
            verbose=False
        )

        if len(results) == 0:
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        det_confs = boxes.conf.cpu().numpy()

        for box, det_conf in zip(xyxy, det_confs):
            x1, y1, x2, y2 = box.tolist()

            cx1, cy1, cx2, cy2 = apply_margin_and_clip(
                x1, y1, x2, y2, img_w, img_h, margin_ratio=CROP_MARGIN_RATIO
            )

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = image.crop((cx1, cy1, cx2, cy2))

            category_id, cls_conf = classify_crop(
                crop_img=crop,
                classifier=classifier,
                transform=cls_transform,
                device=device,
                idx_to_class=idx_to_class
            )

            bbox_x = int(round(x1))
            bbox_y = int(round(y1))
            bbox_w = int(round(x2 - x1))
            bbox_h = int(round(y2 - y1))

            score = float(det_conf * cls_conf)

            rows.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "score": score,
            })
            annotation_id += 1

    submission_df = pd.DataFrame(rows, columns=[
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ])

    submission_df.to_csv(SAVE_CSV, index=False)

    print(f"저장 완료: {SAVE_CSV}")
    print(f"총 예측 객체 수: {len(submission_df)}")
    print(submission_df.head())


if __name__ == "__main__":
    main()