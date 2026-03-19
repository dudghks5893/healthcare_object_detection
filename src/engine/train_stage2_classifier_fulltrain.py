from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import pandas as pd

from src.utils import (
    get_device,
    set_seed,
    build_class_mapping,
    save_class_mapping_json,
)
from src.datasets import PillCropDataset
from src.models import ResNetClassifierModel

"""
    실행 순서: 7

    직접 실행:
    python -m src.engine.train_stage2_classifier_fulltrain

    [역할]
    Stage2 분류 모델을 학습하는 단계.
    Stage1 detector가 생성한 알약 crop 이미지를 입력으로 받아,
    각 crop이 어떤 category_id(알약 종류)인지 분류하는 모델을 학습한다.

    [하는 일]
    - full-train crop metadata CSV를 읽어 DataLoader 생성
    - class_to_idx / idx_to_class 매핑 생성
    - ResNetClassifierModel 기반 분류 모델 학습
    - train_loss 기준으로 best.pt 저장
    - last.pt, class 매핑 json, 하이퍼파라미터 json 저장
    - W&B에 학습 로그 및 모델 artifact 기록

    [결과]
    checkpoints/v1/stage2_classifier/resnet18/
    ├── best.pt
    ├── last.pt
    ├── class_to_idx.json
    ├── idx_to_class.json
    └── hparams.json
"""

FULL_TRAIN_CSV = Path("data/processed/v1/stage2_classifier_crop_dataset/metadata/full_train_crop_labels.csv")
SAVE_DIR = Path("checkpoints/v1/stage2_classifier/resnet18")

MODEL_NAME = "resnet18"
PRETRAINED = False

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
SEED = 42
NUM_WORKERS = 4
PIN_MEMORY = True

def build_transforms(img_size: int = 224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return tf

def build_dataloader(
    csv_path: Path,
    batch_size: int,
    img_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    tf = build_transforms(img_size)
    class_to_idx, idx_to_class = build_class_mapping(csv_path)

    dataset = PillCropDataset(csv_path, class_to_idx=class_to_idx, transform=tf)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dataset, loader, class_to_idx, idx_to_class


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


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def train_stage2_classifier_fulltrain(
    full_train_csv: Path = FULL_TRAIN_CSV,
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
):
    set_seed(seed)
    device = get_device()
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset, loader, class_to_idx, idx_to_class = build_dataloader(
        csv_path=full_train_csv,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = len(class_to_idx)
    model = ResNetClassifierModel(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hparams = {
        "mode": "stage2_v1",
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
        "full_train_csv": str(full_train_csv),
        "save_dir": str(save_dir),
    }

    print("num_classes:", num_classes)
    print("train samples:", len(dataset))
    print("save_dir:", save_dir)

    with wandb.init(
        project="test",
        name=f"stage2_{model_name}_v1",
        config=hparams,
    ) as run:
        best_loss = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, loader, criterion, optimizer, device
            )

            print(
                f"[Epoch {epoch:02d}/{epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            )

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

            if train_loss < best_loss:
                best_loss = train_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

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

        run.summary["best_train_loss"] = best_loss
        run.summary["best_model_path"] = str(best_path)
        run.summary["last_model_path"] = str(last_path)
        run.summary["class_to_idx_path"] = str(class_to_idx_path)
        run.summary["idx_to_class_path"] = str(idx_to_class_path)
        run.summary["hparams_path"] = str(hparams_path)

        artifact = wandb.Artifact("stage2-classifier-fulltrain-v1", type="model")
        artifact.add_file(str(best_path))
        artifact.add_file(str(last_path))
        artifact.add_file(str(class_to_idx_path))
        artifact.add_file(str(idx_to_class_path))
        artifact.add_file(str(hparams_path))
        run.log_artifact(artifact)

        print(f"\nBest train loss: {best_loss:.4f}")
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
            "best_loss": best_loss,
        }


def main():
    output = train_stage2_classifier_fulltrain(
        full_train_csv=FULL_TRAIN_CSV,
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
    )

    print("\n학습 완료")
    print(f"best.pt: {output['best_path']}")
    print(f"last.pt: {output['last_path']}")
    print(f"hparams.json: {output['hparams_path']}")


if __name__ == "__main__":
    main()