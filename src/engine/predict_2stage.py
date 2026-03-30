from pathlib import Path
import json
import re
from collections import defaultdict

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

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
    1. Stage1 detector(YOLO)로 bbox 검출
    2. 검출된 bbox 기준으로 원본 이미지를 crop
    3. crop 이미지를 Stage2 classifier(ResNet)에 넣어 category_id 예측
    4. 최종 submission csv 생성
    5. 예측 결과 시각화 이미지 저장
    6. Stage1 bbox로 잘라낸 crop 이미지를 클래스별로 최대 n개 저장

    [결과]
    outputs/submission/v1/predict_2stage.csv
    outputs/predict/v1/*.png
    outputs/predict/v1/by_class/class_xxxxx/*.png
    outputs/stage1_bbox_by_crop/class_xxxxx/*.png
"""

TEST_IMG_DIR = Path("data/raw/v1/test_images")

DETECTOR_WEIGHT = Path("checkpoints/v1/stage1_detector/yolo11n/baseline/weights/best.pt")
CLASSIFIER_WEIGHT = Path("checkpoints/v1/stage2_classifier/resnet18/best.pt")
CLASSIFIER_DIR = Path("checkpoints/v1/stage2_classifier/resnet18")

SUBMISSION_DIR = Path("outputs/submission/v1")
SAVE_CSV = SUBMISSION_DIR / "predict_2stage.csv"

PREDICT_VIS_DIR = Path("outputs/predict/v1")

# Stage1 bbox로 crop 된 이미지를 클래스별로 저장할 폴더
STAGE1_CROP_SAVE_DIR = Path("outputs/stage1_bbox_by_crop")

DET_IMGSZ = 960
DET_CONF = 0.25
DET_IOU = 0.50
CROP_MARGIN_RATIO = 0.10

SAVE_VIS_LIMIT = 10         # 전체 예측 이미지 저장 개수
SAVE_VIS_PER_CLASS = 5      # 예측 시각화 이미지 클래스별 저장 개수
SAVE_CROP_PER_CLASS = 5     # crop 이미지 클래스별 저장 개수
VIS_FONT_SIZE = 28          # bbox 텍스트 크기
VIS_LINE_WIDTH = 4          # bbox 선 두께


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_classifier_transform(img_size: int = 224):
    """
    Stage2 분류 모델 입력용 transform
    crop 이미지를 classifier 입력 크기로 맞춘 뒤 tensor/normalize 수행
    """
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
    """
    Stage2 분류 모델과 class mapping 정보 로드
    """
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
    """
    Stage2 분류기에서 crop 이미지를 입력받아
    최종 class_id와 confidence를 반환
    """
    x = transform(crop_img).unsqueeze(0).to(device)

    logits = classifier(x)
    probs = F.softmax(logits, dim=1)

    conf, pred_idx = probs.max(dim=1)
    pred_idx = int(pred_idx.item())
    conf = float(conf.item())

    pred_class_id = int(idx_to_class[pred_idx])
    return pred_class_id, conf


def apply_margin_and_clip(x1, y1, x2, y2, img_w, img_h, margin_ratio=0.10):
    """
    Stage1 bbox에 약간의 margin을 추가하고,
    이미지 바깥으로 나가지 않도록 clip 처리
    """
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
    """
    파일명에서 image_id 추출
    """
    stem = Path(file_name).stem

    if stem.isdigit():
        return int(stem)

    numbers = re.findall(r"\d+", stem)
    if not numbers:
        raise ValueError(f"image_id를 추출할 수 없습니다: {file_name}")

    return int(numbers[0])


def load_font(font_size: int):
    """
    사용 가능한 truetype 폰트를 우선 사용하고,
    없으면 기본 폰트로 fallback
    """
    candidate_fonts = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for font_path in candidate_fonts:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                pass

    return ImageFont.load_default()


def get_text_box(draw, text, font, x, y, padding=4):
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    return (
        left - padding,
        top - padding,
        right + padding,
        bottom + padding,
    )


def generate_class_color(class_id: int) -> tuple[int, int, int]:
    """
    class_id가 같으면 항상 같은 색을 반환
    너무 어둡거나 너무 밝지 않도록 범위를 조정
    """
    r = (class_id * 37) % 180 + 50
    g = (class_id * 67) % 180 + 50
    b = (class_id * 97) % 180 + 50
    return (r, g, b)


def draw_predictions(
    image,
    predictions,
    class_color_map: dict[int, tuple[int, int, int]],
    font_size: int = 28,
    line_width: int = 4,
):
    """
    원본 이미지 위에 최종 예측 bbox와 class/score를 그려서 시각화
    """
    draw = ImageDraw.Draw(image)
    font = load_font(font_size)

    for pred in predictions:
        x, y, w, h = pred["bbox"]
        cls_id = int(pred["category_id"])
        score = pred["score"]

        x2 = x + w
        y2 = y + h

        color = class_color_map.get(cls_id, generate_class_color(cls_id))

        draw.rectangle([x, y, x2, y2], outline=color, width=line_width)

        text = f"{cls_id} ({score:.2f})"
        text_x = x
        text_y = max(0, y - font_size - 8)

        bg_box = get_text_box(draw, text, font, text_x, text_y, padding=4)
        draw.rectangle(bg_box, fill=color)
        draw.text((text_x, text_y), text, fill="white", font=font)

    return image


def save_classwise_visualizations(
    image: Image.Image,
    image_predictions: list[dict],
    file_name: str,
    predict_vis_dir: Path,
    class_vis_counter: dict,
    save_vis_per_class: int,
    font_size: int,
    line_width: int,
    class_color_map: dict[int, tuple[int, int, int]],
):
    """
    최종 예측 결과를 원본 이미지 위에 그린 뒤
    클래스별로 n개씩 저장
    """
    class_ids_in_image = sorted({pred["category_id"] for pred in image_predictions})

    for class_id in class_ids_in_image:
        if class_vis_counter[class_id] >= save_vis_per_class:
            continue

        class_preds = [pred for pred in image_predictions if pred["category_id"] == class_id]

        vis_img = image.copy()
        vis_img = draw_predictions(
            vis_img,
            class_preds,
            class_color_map=class_color_map,
            font_size=font_size,
            line_width=line_width,
        )

        class_dir = predict_vis_dir / f"class_{class_id}"
        class_dir.mkdir(parents=True, exist_ok=True)

        save_index = class_vis_counter[class_id] + 1
        save_path = class_dir / f"{Path(file_name).stem}_class{class_id}_{save_index:03d}.png"
        vis_img.save(save_path)

        class_vis_counter[class_id] += 1


def save_stage1_crop_by_class(
    crop_img: Image.Image,
    category_id: int,
    file_name: str,
    annotation_id: int,
    save_root: Path,
    crop_save_counter: dict,
    save_crop_per_class: int,
    score: float,
):
    """
    Stage1 bbox를 기준으로 잘라낸 crop 이미지를
    Stage2의 최종 예측 class 기준으로 클래스별 n개씩 저장

    역할:
    - Stage1이 실제로 어떤 영역을 crop해서 Stage2에 넘겼는지 눈으로 확인
    - 오분류가 bbox 문제인지, crop 문제인지, classifier 문제인지 분석할 때 사용
    """
    if crop_save_counter[category_id] >= save_crop_per_class:
        return

    class_dir = save_root / f"class_{category_id}"
    class_dir.mkdir(parents=True, exist_ok=True)

    save_index = crop_save_counter[category_id] + 1
    save_path = class_dir / (
        f"{Path(file_name).stem}_ann{annotation_id}_"
        f"class{category_id}_{save_index:03d}_{score:.3f}.png"
    )

    crop_img.save(save_path)
    crop_save_counter[category_id] += 1


def predict_2stage(
    test_img_dir: Path = TEST_IMG_DIR,
    detector_weight: Path = DETECTOR_WEIGHT,
    classifier_weight: Path = CLASSIFIER_WEIGHT,
    classifier_dir: Path = CLASSIFIER_DIR,
    save_csv: Path = SAVE_CSV,
    predict_vis_dir: Path = PREDICT_VIS_DIR,
    stage1_crop_save_dir: Path = STAGE1_CROP_SAVE_DIR,
    det_imgsz: int = DET_IMGSZ,
    det_conf: float = DET_CONF,
    det_iou: float = DET_IOU,
    crop_margin_ratio: float = CROP_MARGIN_RATIO,
    save_vis_limit: int = SAVE_VIS_LIMIT,
    save_vis_per_class: int = SAVE_VIS_PER_CLASS,
    save_crop_per_class: int = SAVE_CROP_PER_CLASS,
    vis_font_size: int = VIS_FONT_SIZE,
    vis_line_width: int = VIS_LINE_WIDTH,
):
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    predict_vis_dir.mkdir(parents=True, exist_ok=True)
    stage1_crop_save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    if device.type == "cuda":
        yolo_device = "0"
    elif device.type == "mps":
        yolo_device = "mps"
    else:
        yolo_device = "cpu"

    # -----------------------------------------------------------
    # Stage1 detector 로드
    # 역할:
    # - 원본 이미지에서 알약의 위치(bbox)를 찾는 모델
    # -----------------------------------------------------------
    detector = YOLO(str(detector_weight))

    # -----------------------------------------------------------
    # Stage2 classifier 로드
    # 역할:
    # - Stage1 bbox로 잘라낸 crop 이미지를 받아
    #   최종 class_id를 예측하는 모델
    # -----------------------------------------------------------
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
    class_vis_counter = defaultdict(int)
    crop_save_counter = defaultdict(int)
    class_color_map = {}

    for img_path in image_paths:
        file_name = img_path.name
        image_id = extract_image_id(file_name)

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        image_predictions = []

        # -----------------------------------------------------------
        # [Stage1]
        # 원본 이미지를 detector에 넣어서 bbox 예측
        # 여기서 Stage1이 "알약이 어디 있는지" 찾는다
        # -----------------------------------------------------------
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

        # -----------------------------------------------------------
        # [Stage1 bbox 추출]
        # Stage1 detector가 예측한 bbox 좌표와 detector confidence
        # -----------------------------------------------------------
        xyxy = boxes.xyxy.cpu().numpy()
        det_confs = boxes.conf.cpu().numpy()

        for box, det_score in zip(xyxy, det_confs):
            # -------------------------------------------------------
            # [Stage1 -> bbox 좌표 전달]
            # Stage1이 예측한 bbox를 x1, y1, x2, y2 형태로 꺼내는 부분
            # -------------------------------------------------------
            x1, y1, x2, y2 = box.tolist()

            # -------------------------------------------------------
            # bbox 주변에 margin을 조금 추가하고
            # 이미지 범위를 벗어나지 않도록 clip 처리
            # -------------------------------------------------------
            cx1, cy1, cx2, cy2 = apply_margin_and_clip(
                x1, y1, x2, y2,
                img_w, img_h,
                margin_ratio=crop_margin_ratio,
            )

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            # -------------------------------------------------------
            # [Stage2 입력용 crop 생성]
            # Stage1이 찾은 bbox 영역을 실제 이미지에서 잘라냄
            # 이 crop 이미지가 Stage2 classifier로 들어간다
            # -------------------------------------------------------
            crop = image.crop((cx1, cy1, cx2, cy2))

            # -------------------------------------------------------
            # [Stage2 분류]
            # crop 이미지를 Stage2 classifier에 넣어
            # 최종 category_id와 classifier confidence를 얻음
            # -------------------------------------------------------
            category_id, cls_score = classify_crop(
                crop_img=crop,
                classifier=classifier,
                transform=cls_transform,
                device=device,
                idx_to_class=idx_to_class,
            )

            if category_id not in class_color_map:
                class_color_map[category_id] = generate_class_color(category_id)

            bbox_x = int(round(x1))
            bbox_y = int(round(y1))
            bbox_w = int(round(x2 - x1))
            bbox_h = int(round(y2 - y1))

            # -------------------------------------------------------
            # 최종 score
            # detector confidence와 classifier confidence를 곱해서 사용
            # -------------------------------------------------------
            score = float(det_score * cls_score)

            # -------------------------------------------------------
            # Stage1 bbox 기반 crop 이미지를
            # Stage2 최종 class 기준으로 클래스별 5개씩 저장
            # -------------------------------------------------------
            save_stage1_crop_by_class(
                crop_img=crop,
                category_id=category_id,
                file_name=file_name,
                annotation_id=annotation_id,
                save_root=stage1_crop_save_dir,
                crop_save_counter=crop_save_counter,
                save_crop_per_class=save_crop_per_class,
                score=score,
            )

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

        # 전체 예측 이미지 일부 저장
        if saved_vis_count < save_vis_limit and len(image_predictions) > 0:
            vis_img = image.copy()
            vis_img = draw_predictions(
                vis_img,
                image_predictions,
                class_color_map=class_color_map,
                font_size=vis_font_size,
                line_width=vis_line_width,
            )

            save_path = predict_vis_dir / f"{Path(file_name).stem}_pred.png"
            vis_img.save(save_path)
            saved_vis_count += 1

        # 클래스별 예측 시각화 저장
        if len(image_predictions) > 0:
            save_classwise_visualizations(
                image=image,
                image_predictions=image_predictions,
                file_name=file_name,
                predict_vis_dir=predict_vis_dir / "by_class",
                class_vis_counter=class_vis_counter,
                save_vis_per_class=save_vis_per_class,
                font_size=vis_font_size,
                line_width=vis_line_width,
                class_color_map=class_color_map,
            )

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
    print(f"Stage1 crop 저장 폴더: {stage1_crop_save_dir}")
    print(f"총 예측 객체 수: {len(submission_df)}")
    print(f"전체 시각화 저장 개수 제한: {save_vis_limit}")
    print(f"클래스별 시각화 저장 개수 제한: {save_vis_per_class}")
    print(f"클래스별 crop 저장 개수 제한: {save_crop_per_class}")

    if len(submission_df) > 0:
        print(submission_df.head())

    return {
        "save_csv": save_csv,
        "predict_vis_dir": predict_vis_dir,
        "stage1_crop_save_dir": stage1_crop_save_dir,
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
        stage1_crop_save_dir=STAGE1_CROP_SAVE_DIR,
        det_imgsz=DET_IMGSZ,
        det_conf=DET_CONF,
        det_iou=DET_IOU,
        crop_margin_ratio=CROP_MARGIN_RATIO,
        save_vis_limit=SAVE_VIS_LIMIT,
        save_vis_per_class=SAVE_VIS_PER_CLASS,
        save_crop_per_class=SAVE_CROP_PER_CLASS,
        vis_font_size=VIS_FONT_SIZE,
        vis_line_width=VIS_LINE_WIDTH,
    )

    print("\n추론 완료")
    print(f"submission file: {output['save_csv']}")
    print(f"predict image dir: {output['predict_vis_dir']}")
    print(f"stage1 crop save dir: {output['stage1_crop_save_dir']}")
    print(f"num_predictions: {output['num_predictions']}")


if __name__ == "__main__":
    main()