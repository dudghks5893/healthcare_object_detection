import json                      # JSON annotation 파일 읽기용
import random                    # 시드 고정용
import pandas as pd              # CSV 저장용
from pathlib import Path         # 경로 처리용
import os                        # 운영체제 관련 기능 사용
from datetime import datetime    # 저장 파일명에 시간 붙일 때 사용
from collections import Counter  # 클래스 분포 확인용

import kagglehub                 # Kaggle 대회 데이터 다운로드용
import torch                     # PyTorch 메인 라이브러리
import wandb                     # 실험 로그 시각화/기록용
from PIL import Image            # 이미지 로드용
from torchvision.transforms import v2  # torchvision transform
from torch.utils.data import Dataset, DataLoader  # 데이터셋 / 데이터로더
from sklearn.model_selection import train_test_split  # train/valid 분리
from torch.optim.lr_scheduler import MultiStepLR      # learning rate scheduler
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # detection mAP 계산

from torchvision.models.detection import ssd300_vgg16  # SSD 모델
from torchvision.models import VGG16_Weights            # backbone pretrained weight

# dataset_class.py에서 가져오는 것들
from src.datasets.dataset_class import (
    PillDetectionDataset,              # 학습용 detection dataset
    get_torchvision_train_transform,   # train용 증강/전처리
    get_torchvision_valid_transform,   # valid용 전처리
    detection_collate_fn,              # detection용 collate_fn
)

"""
실행 방법:
프로젝트 루트에서

python -m src.models.normal_SSD
"""


def find_data_root(download_root: str):
    """
    KaggleHub가 다운로드한 폴더 안에서 실제 데이터 루트를 찾는 함수

    예:
    download_root/
      sprint_ai_project1_data/
        train_images/
        train_annotations/
        test_images/

    sprint_ai_project1_data가 있으면 그걸 실제 데이터 루트로 사용
    """
    root = Path(download_root)
    candidate = root / "sprint_ai_project1_data"
    if candidate.exists():
        return candidate
    return root


def get_image_and_annotation_dirs(data_root: Path):
    """
    train/test 이미지 폴더와 annotation 폴더 위치를 가져오는 함수
    폴더가 없으면 바로 에러를 내서 경로 문제를 빨리 찾을 수 있게 함
    """
    image_dir = data_root / "train_images"
    annotation_dir = data_root / "train_annotations"
    test_dir = data_root / "test_images"

    if not image_dir.exists():
        raise FileNotFoundError(f"train_images 폴더 없음: {image_dir}")
    if not annotation_dir.exists():
        raise FileNotFoundError(f"train_annotations 폴더 없음: {annotation_dir}")

    print(f"[INFO] image_dir: {image_dir}")
    print(f"[INFO] annotation_dir: {annotation_dir}")
    print(f"[INFO] test_dir: {test_dir}")

    return image_dir, annotation_dir, test_dir


def build_annotation_map(annotation_dir: Path):
    """
    annotation 폴더 아래의 모든 json 파일을 찾아서
    file_stem -> json_path 형태의 dict로 만드는 함수

    예:
    "abc123" -> ".../abc123.json"

    Dataset에서 file_name으로 annotation을 바로 찾기 쉽게 하기 위해 사용
    """
    annotation_map = {}

    for json_path in annotation_dir.rglob("*.json"):
        annotation_map[json_path.stem] = str(json_path)

    if len(annotation_map) == 0:
        raise ValueError(f"annotation json 파일을 찾지 못했습니다: {annotation_dir}")

    return annotation_map


def build_file_list(image_dir: Path, annotation_map: dict):
    """
    실제 학습에 사용할 파일 목록 생성

    - 이미지 폴더에 있는 파일 stem
    - annotation_map에 있는 json stem

    이 둘의 교집합만 사용해서
    '이미지도 있고 annotation도 있는 샘플'만 학습하게 함
    """
    image_paths = (
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg"))
    )
    image_stems = {p.stem for p in image_paths}
    ann_stems = set(annotation_map.keys())

    file_list = sorted(list(image_stems & ann_stems))

    if len(file_list) == 0:
        raise ValueError("공통 stem을 가진 image/json 파일이 없습니다.")

    return file_list


def build_class_mapping(annotation_map: dict):
    """
    클래스 이름과 모델 내부 인덱스를 연결하는 함수

    반환:
    - class_to_idx: 모델 학습용 name -> internal index
    - idx_to_class: internal index -> name
    - name_to_original_id: name -> 원본 category_id

    왜 필요하냐:
    - 모델은 문자열 클래스명이 아니라 숫자 라벨만 처리 가능
    - submission 저장 시에는 다시 원본 category_id로 바꿔줘야 함
    """
    name_to_original_id = {}

    for _, json_path in annotation_map.items():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        for cat in categories:
            cat_id = cat.get("id")
            cat_name = cat.get("name")

            if cat_id is None or cat_name is None:
                continue

            # 같은 클래스 이름은 같은 원본 id를 가진다고 가정
            if cat_name not in name_to_original_id:
                name_to_original_id[cat_name] = cat_id

    class_names = sorted(name_to_original_id.keys())

    if len(class_names) == 0:
        raise ValueError("class를 하나도 찾지 못했습니다. JSON 구조를 확인하세요.")

    # 모델은 1부터 시작하는 내부 라벨을 사용
    # 0은 background로 비워두는 관행을 따름
    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    return class_to_idx, idx_to_class, name_to_original_id


def inspect_train_distribution(train_files, annotation_map):
    """
    학습 데이터의 클래스 분포를 출력하는 함수

    왜 필요하냐:
    - 특정 클래스가 너무 많으면 모델이 그 클래스로 쏠릴 수 있음
    - 지금처럼 category_id가 한 클래스만 나오는 문제를 분석할 때 유용
    """
    counter = Counter()

    for file_name in train_files:
        json_path = annotation_map[file_name]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # category_id -> class_name 매핑
        cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}

        for ann in data.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id in cat_map:
                counter[cat_map[cat_id]] += 1

    print("[INFO] train class distribution (top 20):")
    for name, count in counter.most_common(20):
        print(f"  {name}: {count}")


def build_model(num_classes: int):
    """
    SSD 모델 생성

    - weights=None:
      COCO detection head 전체 pretrained를 쓰지 않음
      (우리 데이터셋 클래스 수가 COCO와 다르기 때문)

    - weights_backbone=...:
      backbone feature extractor는 ImageNet pretrained 사용
    """
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
        num_classes=num_classes,
    )
    return model


class TestImageDataset(Dataset):
    """
    submission 생성용 테스트 이미지 Dataset

    학습용 Dataset과 달리 annotation이 없고
    이미지와 파일명만 반환함
    """
    def __init__(self, file_list, image_dir, transform=None):
        self.file_list = file_list
        self.image_dir = image_dir

        # 테스트용 기본 전처리
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        # 확장자가 png/jpg/jpeg 중 무엇인지 자동 탐색
        possible_paths = [
            self.image_dir / f"{file_name}.png",
            self.image_dir / f"{file_name}.jpg",
            self.image_dir / f"{file_name}.jpeg",
        ]

        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break

        if img_path is None:
            raise FileNotFoundError(f"테스트 이미지 없음: {file_name}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 테스트셋은 label이 없으므로 file_name을 함께 반환
        return image, file_name


def build_test_file_list(test_dir: Path):
    """
    test 폴더 아래 이미지 파일 목록을 읽어서
    stem 기준 정렬된 리스트를 만드는 함수
    """
    image_paths = (
        list(test_dir.glob("*.png")) +
        list(test_dir.glob("*.jpg")) +
        list(test_dir.glob("*.jpeg"))
    )
    return sorted([p.stem for p in image_paths])


def test_collate_fn(batch):
    """
    테스트셋용 collate 함수

    detection은 기본 collate가 불편할 수 있으므로
    이미지 리스트 / 파일명 리스트로 분리해서 반환
    """
    images, file_names = zip(*batch)
    return list(images), list(file_names)


@torch.no_grad()
def save_submission_csv(
    model,
    test_loader,
    device,
    csv_path,
    idx_to_class,
    name_to_original_id,
    score_thr=0.5,
):
    """
    테스트셋 예측 결과를 submission.csv로 저장하는 함수

    저장 형식:
    annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
    """
    model.eval()

    rows = []
    annotation_id = 1
    debug_printed = False

    for images, file_names in test_loader:
        # 이미지를 GPU/CPU device로 이동
        images = [img.to(device) for img in images]

        # SSD 모델 예측
        preds = model(images)

        for file_name, pred in zip(file_names, preds):
            boxes = pred["boxes"].detach().cpu().tolist()
            labels = pred["labels"].detach().cpu().tolist()
            scores = pred["scores"].detach().cpu().tolist()

            # 첫 배치에서 디버깅용 출력
            if not debug_printed:
                print("[DEBUG] pred labels sample:", labels[:20])
                print("[DEBUG] idx_to_class sample:", list(idx_to_class.items())[:10])
                print("[DEBUG] name_to_original_id sample:", list(name_to_original_id.items())[:10])
                debug_printed = True

            # file_name이 숫자면 int로 저장, 아니면 문자열 그대로
            try:
                image_id = int(file_name)
            except ValueError:
                image_id = file_name

            # 예측 박스 하나씩 CSV row로 변환
            for box, label, score in zip(boxes, labels, scores):
                if score < score_thr:
                    continue

                xmin, ymin, xmax, ymax = box
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                # 모델 내부 라벨 -> 클래스 이름
                class_name = idx_to_class.get(int(label), None)
                if class_name is None:
                    continue

                # 클래스 이름 -> 원본 category_id
                original_category_id = name_to_original_id.get(class_name, None)
                if original_category_id is None:
                    continue

                rows.append({
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(original_category_id),
                    "bbox_x": float(xmin),
                    "bbox_y": float(ymin),
                    "bbox_w": float(bbox_w),
                    "bbox_h": float(bbox_h),
                    "score": float(score),
                })

                annotation_id += 1

    # DataFrame으로 만들어 csv 저장
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

    submission_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"submission CSV 저장 완료: {csv_path}")


def targets_to_wandb_boxes(target, idx_to_class):
    """
    GT(target)를 W&B boxes 포맷으로 바꾸는 함수

    왜 필요하냐:
    - validation 샘플을 W&B에 업로드할 때
      GT 박스를 시각화하기 위해 사용
    """
    box_data = []
    boxes = target["boxes"].detach().cpu().tolist()
    labels = target["labels"].detach().cpu().tolist()

    for box, cls_id in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        box_data.append({
            "position": {
                "minX": xmin,
                "minY": ymin,
                "maxX": xmax,
                "maxY": ymax,
            },
            "class_id": int(cls_id),
            "box_caption": idx_to_class.get(int(cls_id), str(cls_id)),
            "domain": "pixel",
        })

    return {
        "ground_truth": {
            "box_data": box_data,
            "class_labels": idx_to_class,
        }
    }


def predictions_to_wandb_boxes(pred, idx_to_class, score_thr=0.3):
    """
    모델 예측을 W&B boxes 포맷으로 바꾸는 함수

    score threshold보다 낮은 예측은 시각화에서 제외
    """
    box_data = []
    boxes = pred["boxes"].detach().cpu().tolist()
    labels = pred["labels"].detach().cpu().tolist()
    scores = pred["scores"].detach().cpu().tolist()

    for box, cls_id, score in zip(boxes, labels, scores):
        if score < score_thr:
            continue

        xmin, ymin, xmax, ymax = box
        box_data.append({
            "position": {
                "minX": xmin,
                "minY": ymin,
                "maxX": xmax,
                "maxY": ymax,
            },
            "class_id": int(cls_id),
            "box_caption": f"{idx_to_class.get(int(cls_id), cls_id)} {score:.2f}",
            "scores": {"score": float(score)},
            "domain": "pixel",
        })

    return {
        "predictions": {
            "box_data": box_data,
            "class_labels": idx_to_class,
        }
    }


@torch.no_grad()
def evaluate_map(model, valid_loader, device):
    """
    validation set에서 mAP 계산

    torchmetrics의 MeanAveragePrecision 사용
    """
    model.eval()
    metric = MeanAveragePrecision()

    for images, targets in valid_loader:
        images = [img.to(device) for img in images]
        targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

        preds = model(images)
        metric.update(preds, targets_gpu)

    return metric.compute()


@torch.no_grad()
def log_validation_samples(model, valid_loader, device, idx_to_class, step, max_images=4):
    """
    validation 샘플 몇 장을 W&B에 업로드하는 함수

    - GT 박스
    - 예측 박스
    를 한 이미지에 같이 표시해서
    모델이 실제로 무엇을 예측하는지 확인할 수 있게 해줌
    """
    model.eval()

    images, targets = next(iter(valid_loader))
    images = [img.to(device) for img in images]
    preds = model(images)

    wandb_images = []

    for i in range(min(len(images), max_images)):
        img_cpu = images[i].detach().cpu()
        gt_boxes = targets_to_wandb_boxes(targets[i], idx_to_class)
        pred_boxes = predictions_to_wandb_boxes(preds[i], idx_to_class, score_thr=0.3)

        merged_boxes = {}
        merged_boxes.update(gt_boxes)
        merged_boxes.update(pred_boxes)

        wandb_images.append(
            wandb.Image(
                img_cpu,
                boxes=merged_boxes,
                caption=f"val_sample_{i}"
            )
        )

    wandb.log({"val_samples": wandb_images}, step=step)


def main():
    """
    전체 학습 파이프라인 메인 함수

    흐름:
    1. wandb 시작
    2. 데이터 다운로드 및 경로 설정
    3. annotation/json 정리
    4. class mapping 생성
    5. train/valid split
    6. dataset/dataloader 생성
    7. 모델/optimizer/scheduler 생성
    8. epoch 학습
    9. validation 평가 및 시각화
    10. best model 저장
    11. history csv 저장
    12. submission csv 저장
    """
    wandb.init(
        project="pill-ssd",
        name="ssd300_vgg16_augmented",
        config={
            "epochs": 20,
            "batch_size": 8,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "valid_ratio": 0.2,
            "seed": 42,
            "milestones": [12, 16],
            "gamma": 0.1,
        }
    )

    cfg = wandb.config

    # 실험 재현성을 위해 시드 고정
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # CUDA 가능하면 GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": str(device)}, allow_val_change=True)

    # Kaggle 데이터 다운로드
    download_root = kagglehub.competition_download("ai09-level1-project")
    print("download_root:", download_root)

    # 실제 데이터 루트 / 폴더 경로 찾기
    data_root = find_data_root(download_root)
    image_dir, annotation_dir, test_dir = get_image_and_annotation_dirs(data_root)

    # annotation map / file list / class mapping 구성
    annotation_map = build_annotation_map(annotation_dir)
    file_list = build_file_list(image_dir, annotation_map)
    class_to_idx, idx_to_class, name_to_original_id = build_class_mapping(annotation_map)

    print("class_to_idx:", class_to_idx)
    print("num_classes(with background):", len(class_to_idx) + 1)
    print("idx_to_class sample:", list(idx_to_class.items())[:10])
    print("name_to_original_id sample:", list(name_to_original_id.items())[:10])

    # train / valid 분리
    train_files, valid_files = train_test_split(
        file_list,
        test_size=cfg.valid_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )

    # 클래스 분포 확인
    inspect_train_distribution(train_files, annotation_map)

    num_classes = len(class_to_idx) + 1

    # wandb config에 데이터 정보 기록
    wandb.config.update({
        "num_train": len(train_files),
        "num_valid": len(valid_files),
        "num_classes_with_background": num_classes,
        "class_to_idx": class_to_idx,
    }, allow_val_change=True)

    # train dataset
    train_dataset = PillDetectionDataset(
        file_list=train_files,
        image_dir=str(image_dir),
        annotation_map=annotation_map,
        class_to_idx=class_to_idx,
        transform=get_torchvision_train_transform(),
        model_type="ssd",
    )

    # valid dataset
    valid_dataset = PillDetectionDataset(
        file_list=valid_files,
        image_dir=str(image_dir),
        annotation_map=annotation_map,
        class_to_idx=class_to_idx,
        transform=get_torchvision_valid_transform(),
        model_type="ssd",
    )

    # dataloader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    # 모델 생성
    model = build_model(num_classes=num_classes)
    model.to(device)

    # optimizer 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # scheduler 설정
    scheduler = MultiStepLR(
        optimizer,
        milestones=list(cfg.milestones),
        gamma=cfg.gamma
    )

    best_map50 = -1.0   # best checkpoint 저장 기준
    history = []        # epoch별 기록 저장용
    global_step = 0     # wandb step

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        # -------------------------
        # 1 epoch 학습
        # -------------------------
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # SSD는 loss dict 반환
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # batch 단위 wandb log
            log_dict = {
                "train/loss_total": loss.item(),
                "train/epoch": epoch + 1,
                "train/batch": batch_idx + 1,
                "train/lr": optimizer.param_groups[0]["lr"],
            }

            # classification/bbox loss 등 세부 loss 기록
            for k, v in loss_dict.items():
                log_dict[f"train/{k}"] = float(v.item())

            wandb.log(log_dict, step=global_step)

        # epoch 끝나면 scheduler step
        scheduler.step()

        epoch_loss = running_loss / max(len(train_loader), 1)
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/epoch_loss": epoch_loss,
            },
            step=global_step
        )

        # -------------------------
        # validation 평가
        # -------------------------
        results = evaluate_map(model, valid_loader, device)

        val_map = float(results["map"].item()) if "map" in results else 0.0
        val_map50 = float(results["map_50"].item()) if "map_50" in results else 0.0
        val_mar100 = float(results["mar_100"].item()) if "mar_100" in results else 0.0

        wandb.log(
            {
                "val/map": val_map,
                "val/map_50": val_map50,
                "val/mar_100": val_mar100,
            },
            step=global_step
        )

        # 2 epoch마다 validation 이미지 시각화 업로드
        if (epoch + 1) % 2 == 0:
            log_validation_samples(
                model=model,
                valid_loader=valid_loader,
                device=device,
                idx_to_class=idx_to_class,
                step=global_step,
                max_images=4,
            )

        # checkpoint 폴더 생성
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)

        # latest checkpoint 저장
        latest_ckpt = ckpt_dir / "ssd_latest.pth"
        torch.save(model.state_dict(), latest_ckpt)

        # best mAP50 기준 best model 저장
        if val_map50 > best_map50:
            best_map50 = val_map50
            best_ckpt = ckpt_dir / "ssd_best_map50.pth"
            torch.save(model.state_dict(), best_ckpt)
            wandb.save(str(best_ckpt))

        print(
            f"[Epoch {epoch + 1}/{cfg.epochs}] "
            f"train_loss={epoch_loss:.4f}, "
            f"val_map={val_map:.4f}, "
            f"val_map50={val_map50:.4f}"
        )

        # history csv용 epoch 기록
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_map": val_map,
            "val_map50": val_map50,
            "val_mar100": val_mar100,
            "lr": optimizer.param_groups[0]["lr"],
        })

    # 저장 파일명에 시간 붙이기
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_dir = Path.home() / "Downloads"

    # -------------------------
    # 학습 기록 CSV 저장
    # -------------------------
    history_df = pd.DataFrame(history)
    history_csv_path = download_dir / f"training_history_{timestamp}.csv"
    history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
    print(f"학습 기록 CSV 저장 완료: {history_csv_path}")

    # -------------------------
    # submission CSV 저장
    # -------------------------
    test_files = build_test_file_list(test_dir)

    test_dataset = TestImageDataset(
        file_list=test_files,
        image_dir=test_dir,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_collate_fn,
    )

    submission_csv_path = download_dir / f"submission_{timestamp}.csv"

    save_submission_csv(
        model=model,
        test_loader=test_loader,
        device=device,
        csv_path=submission_csv_path,
        idx_to_class=idx_to_class,
        name_to_original_id=name_to_original_id,
        score_thr=0.5,
    )

    print(f"Submission 저장 완료: {submission_csv_path}")

    # 실험 종료
    wandb.finish()


if __name__ == "__main__":
    main()