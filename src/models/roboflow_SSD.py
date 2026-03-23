import os
import json
import zipfile
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import ParameterGrid, train_test_split
from torchvision.transforms import v2
from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.datasets.dataset_class import (
    PillDetectionDataset,
    detection_collate_fn,
)

"""
실행 방법:
프로젝트 루트에서

python -m src.models.roboflow_SSD
"""


# -------------------------------------------------
# 1. 압축 해제 / 경로 처리
# -------------------------------------------------
def extract_zip_dataset(zip_path: str, extract_root: str = "data/roboflow_extracted"):
    zip_path = Path(zip_path)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"zip 파일을 찾지 못했습니다: {zip_path}")

    dataset_name = zip_path.stem
    extract_dir = extract_root / dataset_name

    if extract_dir.exists():
        print(f"[INFO] 기존 압축 해제 폴더 사용: {extract_dir}")
        return extract_dir

    print(f"[INFO] 압축 해제 중: {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    return extract_dir


def find_roboflow_split_dirs(dataset_root: Path):
    """
    지원 구조:
    1) train / valid / test
    2) train / val / test
    3) train만 존재

    반환:
    - train_dir
    - valid_dir (없으면 None)
    - test_dir (없으면 None)
    """
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"train 폴더를 찾지 못했습니다: {train_dir}")

    final_valid_dir = None
    if valid_dir.exists():
        final_valid_dir = valid_dir
    elif val_dir.exists():
        final_valid_dir = val_dir

    final_test_dir = test_dir if test_dir.exists() else None

    if final_valid_dir is None:
        print("[WARN] valid/val 폴더가 없습니다. train 폴더에서 train/valid를 직접 분할합니다.")

    if final_test_dir is None:
        print("[WARN] test 폴더가 없습니다. submission은 valid split 기준으로 임시 생성합니다.")

    return train_dir, final_valid_dir, final_test_dir


# -------------------------------------------------
# 2. COCO JSON 읽기 / 기존 dataset_class.py 호환용 변환
# -------------------------------------------------
def load_coco_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_name_mappings_from_coco(coco_data: dict):
    """
    반환:
    - class_to_idx: 모델 내부용 name -> index
    - idx_to_class: index -> name
    - name_to_original_id: name -> 원본 category_id
    """
    categories = coco_data.get("categories", [])
    name_to_original_id = {}

    for cat in categories:
        cat_id = cat.get("id")
        cat_name = cat.get("name")
        if cat_id is None or cat_name is None:
            continue
        name_to_original_id[cat_name] = cat_id

    class_names = sorted(name_to_original_id.keys())

    if len(class_names) == 0:
        raise ValueError("categories가 비어 있습니다. COCO JSON 구조를 확인하세요.")

    class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    return class_to_idx, idx_to_class, name_to_original_id


def convert_coco_split_to_annotation_map(split_dir: Path, cache_root: Path):
    """
    Roboflow COCO split의 _annotations.coco.json 을
    기존 PillDetectionDataset이 읽을 수 있게
    이미지별 mini-json으로 쪼개서 annotation_map 생성
    """
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO annotation 파일이 없습니다: {ann_path}")

    coco_data = load_coco_json(ann_path)
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    image_id_to_info = {img["id"]: img for img in images}

    anns_by_image_id = {}
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is None:
            continue
        anns_by_image_id.setdefault(image_id, []).append(ann)

    mini_json_dir = cache_root / split_dir.name
    mini_json_dir.mkdir(parents=True, exist_ok=True)

    annotation_map = {}

    for image_id, img_info in image_id_to_info.items():
        file_name = img_info["file_name"]
        file_stem = Path(file_name).stem

        mini_data = {
            "images": [img_info],
            "annotations": anns_by_image_id.get(image_id, []),
            "categories": categories,
        }

        mini_json_path = mini_json_dir / f"{file_stem}.json"
        with open(mini_json_path, "w", encoding="utf-8") as f:
            json.dump(mini_data, f, ensure_ascii=False)

        annotation_map[file_stem] = str(mini_json_path)

    return split_dir, annotation_map, coco_data


def build_file_list_from_split_dir(image_dir: Path, annotation_map: dict):
    image_paths = (
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg"))
    )
    image_stems = {p.stem for p in image_paths}
    ann_stems = set(annotation_map.keys())

    file_list = sorted(list(image_stems & ann_stems))

    if len(file_list) == 0:
        raise ValueError(f"공통 stem을 가진 이미지/annotation이 없습니다: {image_dir}")

    return file_list


def inspect_train_distribution(file_list, annotation_map):
    counter = Counter()

    for file_name in file_list:
        json_path = annotation_map[file_name]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}

        for ann in data.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id in cat_map:
                counter[cat_map[cat_id]] += 1

    print("[INFO] train distribution (top 20):")
    for name, count in counter.most_common(20):
        print(f"  {name}: {count}")


# -------------------------------------------------
# 3. 증강
# -------------------------------------------------
def get_ssd_train_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        v2.RandomPhotometricDistort(p=0.5),
        v2.RandomZoomOut(fill={0: 0, 1: 0, 2: 0}, side_range=(1.0, 1.3), p=0.3),
    ])


def get_ssd_valid_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


# -------------------------------------------------
# 4. 모델 / 평가
# -------------------------------------------------
def build_model(num_classes: int):
    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
        num_classes=num_classes,
    )
    return model


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
    return {
        "map": float(results["map"].item()) if "map" in results else 0.0,
        "map_50": float(results["map_50"].item()) if "map_50" in results else 0.0,
        "mar_100": float(results["mar_100"].item()) if "mar_100" in results else 0.0,
    }


@torch.no_grad()
def inspect_prediction_distribution(model, valid_loader, device, idx_to_class, max_batches=5):
    model.eval()
    counter = Counter()

    for batch_idx, (images, targets) in enumerate(valid_loader):
        if batch_idx >= max_batches:
            break

        images = [img.to(device) for img in images]
        preds = model(images)

        for pred in preds:
            labels = pred["labels"].detach().cpu().tolist()
            scores = pred["scores"].detach().cpu().tolist()

            for label, score in zip(labels, scores):
                if score >= 0.5:
                    class_name = idx_to_class.get(int(label), str(label))
                    counter[class_name] += 1

    print("[INFO] prediction distribution:")
    for k, v in counter.most_common():
        print(f"  {k}: {v}")


# -------------------------------------------------
# 5. 테스트용 Dataset / submission 생성
# -------------------------------------------------
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
    file_to_image_id=None,
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

            if file_to_image_id is not None and file_name in file_to_image_id:
                image_id = file_to_image_id[file_name]
            else:
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
    print(f"[INFO] submission 저장 완료: {csv_path}")


# -------------------------------------------------
# 6. 단일 trial 학습
# -------------------------------------------------
def run_one_trial(
    trial_id,
    params,
    train_dataset,
    valid_dataset,
    num_classes,
    device,
):
    print(f"\n[TRIAL {trial_id}] params = {params}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )

    model = build_model(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=params["milestones"],
        gamma=params["gamma"],
    )

    history = []
    best_map50 = -1.0
    best_state = None

    for epoch in range(params["epochs"]):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        epoch_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate_map(model, valid_loader, device)

        history.append({
            "trial_id": trial_id,
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_map": metrics["map"],
            "val_map50": metrics["map_50"],
            "val_mar100": metrics["mar_100"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"[TRIAL {trial_id}] "
            f"Epoch {epoch + 1}/{params['epochs']} | "
            f"loss={epoch_loss:.4f} | "
            f"map={metrics['map']:.4f} | "
            f"map50={metrics['map_50']:.4f}"
        )

        if metrics["map_50"] > best_map50:
            best_map50 = metrics["map_50"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return {
        "trial_id": trial_id,
        "params": params,
        "best_map50": best_map50,
        "history": history,
        "best_state": best_state,
    }


# -------------------------------------------------
# 7. 메인
# -------------------------------------------------
def main():
    zip_path = "data/mini_project_dataset.v3i.coco.zip"
    extracted_root = "data/roboflow_extracted"

    param_grid = {
        "batch_size": [4, 8],
        "lr": [1e-4, 5e-4],
        "weight_decay": [1e-4],
        "epochs": [20],
        "milestones": [[12, 16]],
        "gamma": [0.1],
    }

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    dataset_root = extract_zip_dataset(zip_path, extracted_root)
    train_dir, valid_dir, test_dir = find_roboflow_split_dirs(dataset_root)

    cache_root = dataset_root / "_mini_json_cache"

    train_image_dir, train_annotation_map, train_coco = convert_coco_split_to_annotation_map(train_dir, cache_root)

    class_to_idx, idx_to_class, name_to_original_id = build_name_mappings_from_coco(train_coco)
    num_classes = len(class_to_idx) + 1

    print("[INFO] num_classes(with background):", num_classes)
    print("[INFO] class_to_idx sample:", list(class_to_idx.items())[:10])
    print("[INFO] name_to_original_id sample:", list(name_to_original_id.items())[:10])

    all_train_files = build_file_list_from_split_dir(train_image_dir, train_annotation_map)

    if valid_dir is not None:
        valid_image_dir, valid_annotation_map, valid_coco = convert_coco_split_to_annotation_map(valid_dir, cache_root)
        train_files = all_train_files
        valid_files = build_file_list_from_split_dir(valid_image_dir, valid_annotation_map)
    else:
        train_files, valid_files = train_test_split(
            all_train_files,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )
        valid_image_dir = train_image_dir
        valid_annotation_map = train_annotation_map

    inspect_train_distribution(train_files, train_annotation_map)

    train_dataset = PillDetectionDataset(
        file_list=train_files,
        image_dir=str(train_image_dir),
        annotation_map=train_annotation_map,
        class_to_idx=class_to_idx,
        transform=get_ssd_train_transform(),
        model_type="ssd",
    )

    valid_dataset = PillDetectionDataset(
        file_list=valid_files,
        image_dir=str(valid_image_dir),
        annotation_map=valid_annotation_map,
        class_to_idx=class_to_idx,
        transform=get_ssd_valid_transform(),
        model_type="ssd",
    )

    trials = []
    all_history = []

    for trial_id, params in enumerate(ParameterGrid(param_grid), start=1):
        result = run_one_trial(
            trial_id=trial_id,
            params=params,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            num_classes=num_classes,
            device=device,
        )
        trials.append(result)
        all_history.extend(result["history"])

    trials = sorted(trials, key=lambda x: x["best_map50"], reverse=True)
    best_trial = trials[0]

    print("\n[INFO] Best trial")
    print("trial_id:", best_trial["trial_id"])
    print("best_map50:", best_trial["best_map50"])
    print("params:", best_trial["params"])

    best_model = build_model(num_classes=num_classes)
    best_model.load_state_dict(best_trial["best_state"])
    best_model.to(device)
    best_model.eval()

    valid_loader_for_debug = DataLoader(
        valid_dataset,
        batch_size=best_trial["params"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate_fn,
    )
    inspect_prediction_distribution(best_model, valid_loader_for_debug, device, idx_to_class)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / f"ssd_best_{timestamp}.pth"
    torch.save(best_trial["best_state"], best_model_path)
    print(f"[INFO] best model 저장 완료: {best_model_path}")

    history_df = pd.DataFrame(all_history)
    history_csv_path = output_dir / f"training_history_{timestamp}.csv"
    history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] history CSV 저장 완료: {history_csv_path}")

    if test_dir is not None:
        submission_image_dir = test_dir
        submission_files = build_test_file_list(submission_image_dir)
        print("[INFO] test 폴더로 submission 생성")
    else:
        submission_image_dir = valid_image_dir
        submission_files = valid_files
        print("[INFO] test 폴더가 없어 valid split으로 임시 submission 생성")

    test_dataset = TestImageDataset(
        file_list=submission_files,
        image_dir=submission_image_dir,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=best_trial["params"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=test_collate_fn,
    )

    submission_csv_path = output_dir / f"submission_{timestamp}.csv"

    save_submission_csv(
        model=best_model,
        test_loader=test_loader,
        device=device,
        csv_path=submission_csv_path,
        idx_to_class=idx_to_class,
        name_to_original_id=name_to_original_id,
        score_thr=0.5,
    )

    print(f"[INFO] submission CSV 저장 완료: {submission_csv_path}")


if __name__ == "__main__":
    main()