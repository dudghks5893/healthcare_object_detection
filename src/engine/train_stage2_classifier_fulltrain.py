import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from pathlib import Path
import json
import random

import pandas as pd
import torch
import torch.nn as nn
import wandb

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from src.datasets import PillCropDataset
from src.datasets.stage2_classifier_transforms import (
    build_stage2_transforms,
    build_stage2_preview_transform,
)
from src.models import ResNetClassifierModel
from src.utils import (
    build_class_mapping,
    get_device,
    save_class_mapping_json,
    set_fine_tuning,
    set_seed,
)


"""
실행 순서: stage2 fulltrain
직접 실행:
    python -m src.engine.train_stage2_classifier_fulltrain

[역할]
- Stage2 분류 모델을 full-train 방식으로 학습하는 엔진
- train / val split 없이 전체 crop 데이터를 전부 학습에 사용

[입력]
- fulltrain crop metadata csv
  예: data/processed/v2/stage2_classifier_crop_dataset_fulltrain/metadata/fulltrain_crop_labels.csv

[출력]
- best.pt
- last.pt
- class_to_idx.json
- idx_to_class.json
- hparams.json

[특징]
- val set 없이 train 데이터만 사용
- best 모델은 마지막 epoch 기준으로 저장
- early stopping 없음
- class weight는 distribution csv 또는 train csv에서 계산 가능
"""


FULLTRAIN_CSV = Path(
    "data/processed/v2/stage2_classifier_crop_dataset_fulltrain/metadata/fulltrain_crop_labels.csv"
)
CLASS_DIST_CSV = Path("data/processed/v2/class_distribution_v2.csv")
SAVE_DIR = Path("checkpoints/v2/stage2_classifier_fulltrain/resnet50")

MODEL_NAME = "resnet50"
PRETRAINED = True

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
SEED = 42
NUM_WORKERS = 4
PIN_MEMORY = True


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_image_path_column(df: pd.DataFrame):
    candidates = ["crop_path", "image_path", "img_path", "file_path"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "이미지 경로 컬럼을 찾지 못했습니다. "
        "가능 후보: crop_path, image_path, img_path, file_path"
    )


def save_augmentation_preview(
    csv_path: Path,
    save_dir: Path,
    img_size: int = 224,
    augmentation: dict | None = None,
    aug_preview: dict | None = None,
    seed: int = 42,
):
    """
    augmentation preview 이미지 저장
    """
    if not aug_preview or not aug_preview.get("enabled", False):
        return

    num_samples = aug_preview.get("num_samples", 12)
    log_to_wandb = aug_preview.get("log_to_wandb", False)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.empty:
        print("[경고] aug preview용 csv가 비어 있습니다.")
        return

    path_col = find_image_path_column(df)

    random.seed(seed)
    indices = random.sample(range(len(df)), k=min(num_samples, len(df)))

    save_dir.mkdir(parents=True, exist_ok=True)
    preview_tf = build_stage2_preview_transform(
        img_size=img_size,
        augmentation=augmentation,
    )

    wandb_images = []

    for i, idx in enumerate(indices, start=1):
        row = df.iloc[idx]
        img_path = Path(row[path_col])

        if not img_path.exists():
            print(f"[스킵] preview 이미지 없음: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        orig_img = transforms.Resize((img_size, img_size))(img)
        orig_tensor = transforms.ToTensor()(orig_img)
        save_image(orig_tensor, save_dir / f"{i:02d}_orig.png")

        aug_tensor = preview_tf(img)
        aug_path = save_dir / f"{i:02d}_aug.png"
        save_image(aug_tensor, aug_path)

        if log_to_wandb and wandb.run is not None:
            wandb_images.append(
                wandb.Image(str(aug_path), caption=f"sample_{i:02d}")
            )

    if log_to_wandb and wandb_images and wandb.run is not None:
        wandb.log({"stage2/aug_preview": wandb_images})


def build_dataloader(
    csv_path: Path,
    class_to_idx: dict,
    batch_size: int,
    img_size: int,
    train: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    augmentation: dict | None = None,
):
    tf = build_stage2_transforms(
        img_size=img_size,
        train=train,
        augmentation=augmentation,
    )

    dataset = PillCropDataset(
        csv_path=csv_path,
        class_to_idx=class_to_idx,
        transform=tf,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, loader


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    """
    train=True면 학습, False면 평가
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def build_class_weights_from_distribution(
    class_to_idx: dict,
    dist_csv: Path,
) -> torch.Tensor:
    """
    class_distribution csv를 기반으로 class weight 생성
    expected columns:
        - class_id
        - count
    """
    if not dist_csv.exists():
        raise FileNotFoundError(f"class distribution csv가 없습니다: {dist_csv}")

    dist_df = pd.read_csv(dist_csv, encoding="utf-8-sig")

    required_cols = ["class_id", "count"]
    missing_cols = [c for c in required_cols if c not in dist_df.columns]
    if missing_cols:
        raise ValueError(f"class distribution csv에 필수 컬럼이 없습니다: {missing_cols}")

    count_dict = dict(
        zip(
            dist_df["class_id"].astype(str),
            dist_df["count"],
        )
    )
    total = sum(count_dict.values())

    num_classes = len(class_to_idx)
    class_weights = torch.ones(num_classes, dtype=torch.float)

    for class_id, idx in class_to_idx.items():
        class_id = str(class_id)

        if class_id not in count_dict:
            raise KeyError(
                f"class_id={class_id} 가 class distribution csv에 없습니다."
            )

        count = count_dict[class_id]
        class_weights[idx] = total / count

    # 너무 큰 값 완화용 정규화
    class_weights = class_weights / class_weights.mean()
    return class_weights


def build_class_weights_from_train_csv(
    train_csv: Path,
    class_to_idx: dict,
) -> torch.Tensor:
    """
    distribution csv가 없을 때 train csv에서 직접 count를 계산해 class weight 생성
    expected label column:
        - class_id
    """
    df = pd.read_csv(train_csv, encoding="utf-8-sig")

    if "class_id" not in df.columns:
        raise ValueError("train csv에 class_id 컬럼이 없습니다.")

    raw_counts = df["class_id"].astype(str).value_counts().to_dict()
    total = len(df)

    num_classes = len(class_to_idx)
    class_weights = torch.ones(num_classes, dtype=torch.float)

    for class_id, idx in class_to_idx.items():
        class_id = str(class_id)

        if class_id not in raw_counts:
            raise KeyError(f"class_id={class_id} 가 train csv에 없습니다.")

        count = raw_counts[class_id]
        class_weights[idx] = total / count

    class_weights = class_weights / class_weights.mean()
    return class_weights


def train_stage2_classifier_fulltrain(
    wandb_project: str = "test",
    wandb_run_name: str = "train_stage2_classifier_fulltrain",
    fulltrain_csv: Path = FULLTRAIN_CSV,
    class_dist_csv: Path | None = CLASS_DIST_CSV,
    save_dir: Path = SAVE_DIR,
    model_name: str = MODEL_NAME,
    pretrained: bool = PRETRAINED,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    seed: int = SEED,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    augmentation: dict | None = None,
    aug_preview: dict | None = None,
):
    set_seed(seed)
    device = get_device()
    save_dir.mkdir(parents=True, exist_ok=True)

    if not fulltrain_csv.exists():
        raise FileNotFoundError(f"fulltrain csv가 없습니다: {fulltrain_csv}")

    class_to_idx, idx_to_class = build_class_mapping(fulltrain_csv)

    train_dataset, train_loader = build_dataloader(
        csv_path=fulltrain_csv,
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        img_size=img_size,
        train=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        augmentation=augmentation,
    )

    num_classes = len(class_to_idx)

    model = ResNetClassifierModel(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
    ).to(device)

    set_fine_tuning(model, mode="full")

    if class_dist_csv is not None and class_dist_csv.exists():
        class_weights = build_class_weights_from_distribution(
            class_to_idx=class_to_idx,
            dist_csv=class_dist_csv,
        )
        class_weight_source = str(class_dist_csv)
    else:
        class_weights = build_class_weights_from_train_csv(
            train_csv=fulltrain_csv,
            class_to_idx=class_to_idx,
        )
        class_weight_source = f"{fulltrain_csv} (computed)"

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    hparams = {
        "mode": "stage2_fulltrain",
        "model_name": model_name,
        "pretrained": pretrained,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "num_classes": num_classes,
        "seed": seed,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "fulltrain_csv": str(fulltrain_csv),
        "class_dist_csv": str(class_dist_csv) if class_dist_csv is not None else None,
        "class_weight_source": class_weight_source,
        "save_dir": str(save_dir),
        "augmentation": augmentation,
        "aug_preview": aug_preview,
    }

    print("num_classes:", num_classes)
    print("train samples:", len(train_dataset))
    print("save_dir:", save_dir)
    print("class_weight_source:", class_weight_source)

    with wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=hparams,
    ) as run:
        preview_save_dir = save_dir / "aug_preview"
        save_augmentation_preview(
            csv_path=fulltrain_csv,
            save_dir=preview_save_dir,
            img_size=img_size,
            augmentation=augmentation,
            aug_preview=aug_preview,
            seed=seed,
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
            )

            print(
                f"[Epoch {epoch:02d}/{epochs}] "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f}"
            )

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        best_path = save_dir / "best.pt"
        last_path = save_dir / "last.pt"
        hparams_path = save_dir / "hparams.json"

        # fulltrain에서는 마지막 epoch 모델을 best/last 둘 다 저장
        final_state = {k: v.cpu() for k, v in model.state_dict().items()}

        torch.save({
            "model_state_dict": final_state,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "img_size": img_size,
            "model_name": model_name,
            "pretrained": pretrained,
        }, best_path)

        torch.save({
            "model_state_dict": final_state,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "img_size": img_size,
            "model_name": model_name,
            "pretrained": pretrained,
        }, last_path)

        class_to_idx_path, idx_to_class_path = save_class_mapping_json(
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            save_dir=save_dir,
        )

        save_json(hparams, hparams_path)

        run.summary["best_model_path"] = str(best_path)
        run.summary["last_model_path"] = str(last_path)
        run.summary["class_to_idx_path"] = str(class_to_idx_path)
        run.summary["idx_to_class_path"] = str(idx_to_class_path)
        run.summary["hparams_path"] = str(hparams_path)

        print("\n학습 완료")
        print(f"Saved best model to: {best_path}")
        print(f"Saved last model to: {last_path}")
        print(f"Saved class_to_idx to: {class_to_idx_path}")
        print(f"Saved idx_to_class to: {idx_to_class_path}")
        print(f"Saved hparams to: {hparams_path}")

        return {
            "best_path": best_path,
            "last_path": last_path,
            "class_to_idx_path": class_to_idx_path,
            "idx_to_class_path": idx_to_class_path,
            "hparams_path": hparams_path,
        }


def main():
    output = train_stage2_classifier_fulltrain(
        fulltrain_csv=FULLTRAIN_CSV,
        class_dist_csv=CLASS_DIST_CSV,
        save_dir=SAVE_DIR,
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        seed=SEED,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        augmentation=None,
        aug_preview=None,
    )

    print("\n학습 완료")
    print(f"best.pt: {output['best_path']}")
    print(f"last.pt: {output['last_path']}")
    print(f"hparams.json: {output['hparams_path']}")


if __name__ == "__main__":
    main()