import json
import random
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
from collections import Counter

import kagglehub
import torch
import wandb
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights

from src.datasets.dataset_class import (
    PillDetectionDataset,
    get_torchvision_train_transform,
    get_torchvision_valid_transform,
    detection_collate_fn,
)

"""
실행 방법:
프로젝트 루트에서

python -m src.models.normal_SSD
"""


def find_data_root(download_root: str):
    root = Path(download_root)
    candidate = root / "sprint_ai_project1_data"
    if candidate.exists():
        return candidate
    return root


def get_image_and_annotation_dirs(data_root: Path):
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
    annotation_map = {}

    for json_path in annotation_dir.rglob("*.json"):
        annotation_map[json_path.stem] = str(json_path)

    if len(annotation_map) == 0:
        raise ValueError(f"annotation json 파일을 찾지 못했습니다: {annotation_dir}")

    return annotation_map


def build_file_list(image_dir: Path, annotation_map: dict):
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
    반환:
    - class_to_idx: 모델 학습용 name -> internal index
    - idx_to_class: internal index -> name
    - name_to_original_id: name -> 원본 category_id
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

            if cat_name not in name_to_original_id:
                name_to_original_id[cat_name] = cat_id

    class_names = sorted(name_to_original_id.keys())

    if len(class_names) == 0:
        raise ValueError("class를 하나도 찾지 못했습니다. JSON 구조를 확인하세요.")

    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    return class_to_idx, idx_to_class, name_to_original_id


def inspect_train_distribution(train_files, annotation_map):
    counter = Counter()

    for file_name in train_files:
        json_path = annotation_map[file_name]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}

        for ann in data.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id in cat_map:
                counter[cat_map[cat_id]] += 1

    print("[INFO] train class distribution (top 20):")
    for name, count in counter.most_common(20):
        print(f"  {name}: {count}")


def build_model(num_classes: int):
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
        num_classes=num_classes,
    )
    return model


class TestImageDataset(Dataset):
    def __init__(self, file_list, image_dir, transform=None):
        self.file_list = file_list
        self.image_dir = image_dir
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

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

        return image, file_name


def build_test_file_list(test_dir: Path):
    image_paths = (
        list(test_dir.glob("*.png")) +
        list(test_dir.glob("*.jpg")) +
        list(test_dir.glob("*.jpeg"))
    )
    return sorted([p.stem for p in image_paths])


def test_collate_fn(batch):
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
    model.eval()

    rows = []
    annotation_id = 1
    debug_printed = False

    for images, file_names in test_loader:
        images = [img.to(device) for img in images]
        preds = model(images)

        for file_name, pred in zip(file_names, preds):
            boxes = pred["boxes"].detach().cpu().tolist()
            labels = pred["labels"].detach().cpu().tolist()
            scores = pred["scores"].detach().cpu().tolist()

            if not debug_printed:
                print("[DEBUG] pred labels sample:", labels[:20])
                print("[DEBUG] idx_to_class sample:", list(idx_to_class.items())[:10])
                print("[DEBUG] name_to_original_id sample:", list(name_to_original_id.items())[:10])
                debug_printed = True

            try:
                image_id = int(file_name)
            except ValueError:
                image_id = file_name

            for box, label, score in zip(boxes, labels, scores):
                if score < score_thr:
                    continue

                xmin, ymin, xmax, ymax = box
                bbox_w = xmax - xmin
                bbox_h = ymax - ymin

                class_name = idx_to_class.get(int(label), None)
                if class_name is None:
                    continue

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

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": str(device)}, allow_val_change=True)

    download_root = kagglehub.competition_download("ai09-level1-project")
    print("download_root:", download_root)

    data_root = find_data_root(download_root)
    image_dir, annotation_dir, test_dir = get_image_and_annotation_dirs(data_root)

    annotation_map = build_annotation_map(annotation_dir)
    file_list = build_file_list(image_dir, annotation_map)
    class_to_idx, idx_to_class, name_to_original_id = build_class_mapping(annotation_map)

    print("class_to_idx:", class_to_idx)
    print("num_classes(with background):", len(class_to_idx) + 1)
    print("idx_to_class sample:", list(idx_to_class.items())[:10])
    print("name_to_original_id sample:", list(name_to_original_id.items())[:10])

    train_files, valid_files = train_test_split(
        file_list,
        test_size=cfg.valid_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )

    inspect_train_distribution(train_files, annotation_map)

    num_classes = len(class_to_idx) + 1

    wandb.config.update({
        "num_train": len(train_files),
        "num_valid": len(valid_files),
        "num_classes_with_background": num_classes,
        "class_to_idx": class_to_idx,
    }, allow_val_change=True)

    train_dataset = PillDetectionDataset(
        file_list=train_files,
        image_dir=str(image_dir),
        annotation_map=annotation_map,
        class_to_idx=class_to_idx,
        transform=get_torchvision_train_transform(),
        model_type="ssd",
    )

    valid_dataset = PillDetectionDataset(
        file_list=valid_files,
        image_dir=str(image_dir),
        annotation_map=annotation_map,
        class_to_idx=class_to_idx,
        transform=get_torchvision_valid_transform(),
        model_type="ssd",
    )

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

    model = build_model(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=list(cfg.milestones),
        gamma=cfg.gamma
    )

    best_map50 = -1.0
    history = []
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            log_dict = {
                "train/loss_total": loss.item(),
                "train/epoch": epoch + 1,
                "train/batch": batch_idx + 1,
                "train/lr": optimizer.param_groups[0]["lr"],
            }

            for k, v in loss_dict.items():
                log_dict[f"train/{k}"] = float(v.item())

            wandb.log(log_dict, step=global_step)

        scheduler.step()

        epoch_loss = running_loss / max(len(train_loader), 1)
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/epoch_loss": epoch_loss,
            },
            step=global_step
        )

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

        if (epoch + 1) % 2 == 0:
            log_validation_samples(
                model=model,
                valid_loader=valid_loader,
                device=device,
                idx_to_class=idx_to_class,
                step=global_step,
                max_images=4,
            )

        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)

        latest_ckpt = ckpt_dir / "ssd_latest.pth"
        torch.save(model.state_dict(), latest_ckpt)

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

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_map": val_map,
            "val_map50": val_map50,
            "val_mar100": val_mar100,
            "lr": optimizer.param_groups[0]["lr"],
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_dir = Path.home() / "Downloads"

    # 학습 기록 CSV 저장
    history_df = pd.DataFrame(history)
    history_csv_path = download_dir / f"training_history_{timestamp}.csv"
    history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
    print(f"학습 기록 CSV 저장 완료: {history_csv_path}")

    # submission CSV 저장
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

    wandb.finish()


if __name__ == "__main__":
    main()