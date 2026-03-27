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
    EarlyStopping,
    get_calculate_classification_reports,
    get_calculate_metrics
)


"""
    실행 순서: stage2
    직접 실행:
    python -m src.engine.train_stage2_classifier

    [역할]
    Stage2 분류 모델을 train / val 분리 방식으로 학습하는 공용 엔진

    [지원]
    - v1 / v2 공용
    - yaml에서 train_csv, val_csv만 바꿔서 재사용 가능

    [하는 일]
    - train / val crop metadata csv 읽기
    - class_to_idx / idx_to_class 생성
    - train / val dataloader 생성
    - ResNet 기반 분류 모델 학습
    - train / val loss, acc 기록
    - best.pt 저장 (val_loss 기준)
    - class mapping / hparams 저장
    - 선택적으로 augmentation preview 저장
"""


TRAIN_CSV = Path("data/processed/v2/stage2_classifier_crop_dataset/metadata/train_crop_labels.csv")
VAL_CSV = Path("data/processed/v2/stage2_classifier_crop_dataset/metadata/val_crop_labels.csv")
SAVE_DIR = Path("checkpoints/v2/stage2_classifier/resnet50")

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
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

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

        preds = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        # 추가
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_targets, all_preds


def train_stage2_classifier(
    wandb_project: str = "test",
    wandb_run_name: str = "train_stage2_classifier",
    train_csv: Path = TRAIN_CSV,
    val_csv: Path = VAL_CSV,
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

    if not train_csv.exists():
        raise FileNotFoundError(f"train csv가 없습니다: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val csv가 없습니다: {val_csv}")

    class_to_idx, idx_to_class = build_class_mapping(train_csv)

    train_dataset, train_loader = build_dataloader(
        csv_path=train_csv,
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        img_size=img_size,
        train=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        augmentation=augmentation,
    )

    val_dataset, val_loader = build_dataloader(
        csv_path=val_csv,
        class_to_idx=class_to_idx,
        batch_size=batch_size,
        img_size=img_size,
        train=False,
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

    df = pd.read_csv(train_csv)

    raw_counts = df["class_id"].value_counts().to_dict()

    class_weights = torch.ones(num_classes, dtype=torch.float)

    for class_name, idx in class_to_idx.items():
        count = raw_counts[int(class_name)]
        class_weights[idx] = len(df) / count


    # optional (안정화)
    class_weights = class_weights / class_weights.mean()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    hparams = {
        "mode": "stage2_val_split",
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
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "save_dir": str(save_dir),
        "augmentation": augmentation,
        "aug_preview": aug_preview,
    }

    print("num_classes:", num_classes)
    print("train samples:", len(train_dataset))
    print("val samples:", len(val_dataset))
    print("save_dir:", save_dir)

    early_stopping = EarlyStopping(patience=10, min_delta=0.003, mode="min")

    with wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=hparams,
    ) as run:
        preview_save_dir = save_dir / "aug_preview"
        save_augmentation_preview(
            csv_path=train_csv,
            save_dir=preview_save_dir,
            img_size=img_size,
            augmentation=augmentation,
            aug_preview=aug_preview,
            seed=seed,
        )

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            # train 학습
            train_loss, train_acc, train_y, train_pred = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=True,
            )
            train_acc_m, train_recall, train_f1 = get_calculate_metrics(
                train_y,
                train_pred,
            )

            # val 검증
            val_loss, val_acc, val_y, val_pred = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                train=False,
            )
            val_acc_m, val_recall, val_f1 = get_calculate_metrics(
                val_y,
                val_pred,
            )

            print(
                f"[Epoch {epoch:02d}/{epochs}] "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc_m:.4f} "
                f"train_f1={train_f1:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc_m:.4f} "
                f"val_f1={val_f1:.4f}"
            )

            wandb.log({
                "epoch": epoch,

                "train/loss": train_loss,
                "train/acc": train_acc_m,
                "train/recall": train_recall,
                "train/f1": train_f1,

                "val/loss": val_loss,
                "val/acc": val_acc_m,
                "val/recall": val_recall,
                "val/f1": val_f1,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            early_stopping(val_loss)

            if early_stopping.stop:
                print("더이상 학습 개선 진행 불가로 epoch 종료!")
                break

        if best_state is None:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        best_path = save_dir / "best.pt"
        last_path = save_dir / "last.pt"
        hparams_path = save_dir / "hparams.json"

        torch.save({
            "model_state_dict": best_state,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "img_size": img_size,
            "model_name": model_name,
            "pretrained": pretrained,
        }, best_path)

        torch.save({
            "model_state_dict": model.state_dict(),
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

        run.summary["best_val_loss"] = best_val_loss
        run.summary["best_model_path"] = str(best_path)
        run.summary["last_model_path"] = str(last_path)
        run.summary["class_to_idx_path"] = str(class_to_idx_path)
        run.summary["idx_to_class_path"] = str(idx_to_class_path)
        run.summary["hparams_path"] = str(hparams_path)

        print(f"\nBest val loss: {best_val_loss:.4f}")
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
            "best_val_loss": best_val_loss,
        }


def main():
    output = train_stage2_classifier(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
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