from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import wandb
import pandas as pd

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.datasets.stage2_dataset import PillCropDataset

"""
    실행 순서: 7

    [역할]
    crop 이미지를 기반으로 알약 종류를 분류하는 Stage2 classifier를 학습하는 단계

    [하는 일]
    - ResNet 기반 분류 모델 학습
    - class_to_idx 매핑 생성
    - full-train 데이터로 모델 학습

    [결과]
    best.pt / last.pt (classifier weight)
    class_to_idx.json
"""

FULL_TRAIN_CSV = Path("data/processed/stage2_classifier/metadata/full_train_crop_labels.csv")
SAVE_DIR = Path("outputs/stage2_classifier/resnet18_fulltrain")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
SEED = 42


def build_transforms(img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf


def build_class_mapping(csv_path):
    df = pd.read_csv(csv_path)
    class_ids = sorted(df["class_id"].astype(str).unique(), key=lambda x: int(x))
    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(class_ids)}
    idx_to_class = {idx: cls_id for cls_id, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_dataloader(csv_path, batch_size, img_size):
    tf = build_transforms(img_size)
    class_to_idx, idx_to_class = build_class_mapping(csv_path)

    dataset = PillCropDataset(csv_path, class_to_idx=class_to_idx, transform=tf)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return dataset, loader, class_to_idx, idx_to_class


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    set_seed(SEED)
    device = get_device()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    dataset, loader, class_to_idx, idx_to_class = build_dataloader(
        FULL_TRAIN_CSV, BATCH_SIZE, IMG_SIZE
    )

    num_classes = len(class_to_idx)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("num_classes:", num_classes)
    print("train samples:", len(dataset))

    wandb.login()
    with wandb.init(
        project="test",
        name="stage2_resnet18_fulltrain",
        config={
            "mode": "full_train",
            "model": "resnet18",
            "pretrained": False,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "num_classes": num_classes,
            "seed": SEED,
        },
    ) as run:
        best_loss = float("inf")
        best_state = None

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model, loader, criterion, optimizer, device
            )

            print(
                f"[Epoch {epoch:02d}/{EPOCHS}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            )

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

            # full-train에서는 train_loss 기준으로 best 저장
            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        best_path = SAVE_DIR / "best.pt"
        last_path = SAVE_DIR / "last.pt"

        torch.save({
            "model_state_dict": best_state,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "img_size": IMG_SIZE,
        }, best_path)

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "num_classes": num_classes,
            "img_size": IMG_SIZE,
        }, last_path)

        with open(SAVE_DIR / "class_to_idx.json", "w", encoding="utf-8") as f:
            json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

        run.summary["best_train_loss"] = best_loss
        run.summary["best_model_path"] = str(best_path)
        run.summary["last_model_path"] = str(last_path)

        artifact = wandb.Artifact("stage2-classifier-fulltrain", type="model")
        artifact.add_file(str(best_path))
        artifact.add_file(str(last_path))
        run.log_artifact(artifact)

        print(f"Best train loss: {best_loss:.4f}")
        print(f"Saved best model to: {best_path}")
        print(f"Saved last model to: {last_path}")


if __name__ == "__main__":
    main()