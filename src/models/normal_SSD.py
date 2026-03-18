import json
import random
from pathlib import Path

import kagglehub
import torch
import wandb

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
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
    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg"))
    image_stems = {p.stem for p in image_paths}
    ann_stems = set(annotation_map.keys())

    file_list = sorted(list(image_stems & ann_stems))

    if len(file_list) == 0:
        raise ValueError("공통 stem을 가진 image/json 파일이 없습니다.")

    return file_list


def build_class_mapping(annotation_map: dict):
    class_names = set()

    for _, json_path in annotation_map.items():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", [])
        for cat in categories:
            cat_name = cat.get("name")
            if cat_name is not None:
                class_names.add(cat_name)

    class_names = sorted(class_names)

    if len(class_names) == 0:
        raise ValueError("class를 하나도 찾지 못했습니다. JSON 구조를 확인하세요.")

    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    return class_to_idx, idx_to_class


def build_model(num_classes: int):
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
        num_classes=num_classes,
    )
    return model


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

    results = metric.compute()
    return results


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
    class_to_idx, idx_to_class = build_class_mapping(annotation_map)

    print("class_to_idx:", class_to_idx)
    print("num_classes(with background):", len(class_to_idx) + 1)

    train_files, valid_files = train_test_split(
        file_list,
        test_size=cfg.valid_ratio,
        random_state=cfg.seed,
        shuffle=True,
    )

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
            f"[Epoch {epoch+1}/{cfg.epochs}] "
            f"train_loss={epoch_loss:.4f}, "
            f"val_map={val_map:.4f}, "
            f"val_map50={val_map50:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()