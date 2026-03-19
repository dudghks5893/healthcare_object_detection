from pathlib import Path
import json
import re
import random

import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

from src.utils import get_device, load_class_mapping_json
from src.models import ResNetClassifierModel

"""
    실행 순서: 8

    직접 실행:
    python -m src.engine.predict_2stage

    [역할]
    Stage1 Detector + Stage2 Classifier를 결합하여 최종 예측 결과를 생성하는 단계

    [하는 일]
    1. detector로 bbox 검출
    2. bbox 영역 crop
    3. classifier로 category_id 예측
    4. 대회 제출 형식 CSV 생성
    5. 예측 결과 이미지 10장 저장

    [결과]
    outputs/submission/v1/predict_2stage.csv
    outputs/predict/v1/*.png
"""

TEST_IMG_DIR = Path("data/raw/v1/test_images")

DETECTOR_WEIGHT = Path("checkpoints/v1/stage1_detector/yolo11n/baseline/weights/best.pt")
CLASSIFIER_WEIGHT = Path("checkpoints/v1/stage2_classifier/resnet18/best.pt")
CLASSIFIER_DIR = Path("checkpoints/v1/stage2_classifier/resnet18")

SUBMISSION_DIR = Path("outputs/submission/v1")
SAVE_CSV = SUBMISSION_DIR / "predict_2stage.csv"

PREDICT_VIS_DIR = Path("outputs/predict/v1")

DET_IMGSZ = 960
DET_CONF = 0.25
DET_IOU = 0.50
CROP_MARGIN_RATIO = 0.10
SAVE_VIS_LIMIT = 10


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_classifier_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def load_classifier(
    classifier_weight: Path,
    classifier_dir: Path,
    device,
):
    hparams_path = classifier_dir / "hparams.json"
    idx_to_class_path = classifier_dir / "idx_to_class.json"

    hparams = load_json(hparams_path)
    idx_to_class = load_class_mapping_json(idx_to_class_path)

    # json은 key가 문자열이라 int key로 변환
    idx_to_class = {int(k): str(v) for k, v in idx_to_class.items()}

    checkpoint = torch.load(classifier_weight, map_location=device)

    model = ResNetClassifierModel(
        num_classes=checkpoint["num_classes"],
        model_name=checkpoint["model_name"],
        pretrained=checkpoint.get("pretrained", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    img_size = hparams["img_size"]

    return model, idx_to_class, img_size, hparams


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
    stem = Path(file_name).stem

    if stem.isdigit():
        return int(stem)

    numbers = re.findall(r"\d+", stem)
    if not numbers:
        raise ValueError(f"image_id를 추출할 수 없습니다: {file_name}")

    return int(numbers[0])


def draw_predictions(image, predictions):
    """
        predictions: list of dict
        {
            "bbox": (x, y, w, h),
            "category_id": int,
            "score": float
        }
    """
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        x, y, w, h = pred["bbox"]
        cls_id = pred["category_id"]
        score = pred["score"]

        x2 = x + w
        y2 = y + h

        color = tuple(random.randint(0, 255) for _ in range(3))

        draw.rectangle([x, y, x2, y2], outline=color, width=3)
        text = f"{cls_id} ({score:.2f})"

        text_y = max(0, y - 14)
        draw.text((x, text_y), text, fill=color)

    return image


def predict_2stage(
    test_img_dir: Path = TEST_IMG_DIR,
    detector_weight: Path = DETECTOR_WEIGHT,
    classifier_weight: Path = CLASSIFIER_WEIGHT,
    classifier_dir: Path = CLASSIFIER_DIR,
    save_csv: Path = SAVE_CSV,
    predict_vis_dir: Path = PREDICT_VIS_DIR,
    det_imgsz: int = DET_IMGSZ,
    det_conf: float = DET_CONF,
    det_iou: float = DET_IOU,
    crop_margin_ratio: float = CROP_MARGIN_RATIO,
    save_vis_limit: int = SAVE_VIS_LIMIT,
):
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    predict_vis_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    if device.type == "cuda":
        yolo_device = "0"
    elif device.type == "mps":
        yolo_device = "mps"
    else:
        yolo_device = "cpu"

    detector = YOLO(str(detector_weight))

    classifier, idx_to_class, cls_img_size, hparams = load_classifier(
        classifier_weight=classifier_weight,
        classifier_dir=classifier_dir,
        device=device,
    )
    cls_transform = build_classifier_transform(cls_img_size)

    image_paths = sorted(test_img_dir.glob("*"))
    image_paths = [
        p for p in image_paths
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    print(f"test 이미지 수: {len(image_paths)}")
    print(f"detector_weight: {detector_weight}")
    print(f"classifier_weight: {classifier_weight}")
    print(f"classifier_model: {hparams['model_name']}")
    print(f"classifier_img_size: {cls_img_size}")

    rows = []
    annotation_id = 1
    saved_vis_count = 0

    for img_path in image_paths:
        file_name = img_path.name
        image_id = extract_image_id(file_name)

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        image_predictions = []

        results = detector.predict(
            source=str(img_path),
            imgsz=det_imgsz,
            conf=det_conf,
            iou=det_iou,
            device=yolo_device,
            verbose=False,
        )

        if len(results) == 0:
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        det_confs = boxes.conf.cpu().numpy()

        for box, det_score in zip(xyxy, det_confs):
            x1, y1, x2, y2 = box.tolist()

            cx1, cy1, cx2, cy2 = apply_margin_and_clip(
                x1, y1, x2, y2,
                img_w, img_h,
                margin_ratio=crop_margin_ratio,
            )

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = image.crop((cx1, cy1, cx2, cy2))

            category_id, cls_score = classify_crop(
                crop_img=crop,
                classifier=classifier,
                transform=cls_transform,
                device=device,
                idx_to_class=idx_to_class,
            )

            bbox_x = int(round(x1))
            bbox_y = int(round(y1))
            bbox_w = int(round(x2 - x1))
            bbox_h = int(round(y2 - y1))

            score = float(det_score * cls_score)

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

            image_predictions.append({
                "bbox": (bbox_x, bbox_y, bbox_w, bbox_h),
                "category_id": category_id,
                "score": score,
            })

            annotation_id += 1

        if saved_vis_count < save_vis_limit and len(image_predictions) > 0:
            vis_img = image.copy()
            vis_img = draw_predictions(vis_img, image_predictions)

            save_path = predict_vis_dir / f"{Path(file_name).stem}_pred.png"
            vis_img.save(save_path)

            saved_vis_count += 1

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

    submission_df.to_csv(save_csv, index=False)

    print("\n저장 완료")
    print(f"CSV: {save_csv}")
    print(f"시각화 이미지: {predict_vis_dir}")
    print(f"총 예측 객체 수: {len(submission_df)}")
    if len(submission_df) > 0:
        print(submission_df.head())

    return {
        "save_csv": save_csv,
        "predict_vis_dir": predict_vis_dir,
        "num_predictions": len(submission_df),
        "classifier_hparams": hparams,
    }


def main():
    output = predict_2stage(
        test_img_dir=TEST_IMG_DIR,
        detector_weight=DETECTOR_WEIGHT,
        classifier_weight=CLASSIFIER_WEIGHT,
        classifier_dir=CLASSIFIER_DIR,
        save_csv=SAVE_CSV,
        predict_vis_dir=PREDICT_VIS_DIR,
        det_imgsz=DET_IMGSZ,
        det_conf=DET_CONF,
        det_iou=DET_IOU,
        crop_margin_ratio=CROP_MARGIN_RATIO,
        save_vis_limit=SAVE_VIS_LIMIT,
    )

    print("\n추론 완료")
    print(f"submission file: {output['save_csv']}")
    print(f"predict image dir: {output['predict_vis_dir']}")
    print(f"num_predictions: {output['num_predictions']}")


if __name__ == "__main__":
    main()