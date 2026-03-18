from pathlib import Path
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import wandb

from src.utils.device import get_device
from src.utils.seed import set_seed

"""
    실행 방법:
    터미널에 python -m src.engine.train_stage2_classifier 입력
"""

TRAIN_DIR = Path("data/processed/stage2_classifier/train")
VAL_DIR = Path("data/processed/stage2_classifier/val")
SAVE_DIR = Path("outputs/stage2_classifier/resnet18_baseline")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
SEED = 42


def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_dataloaders(train_dir, val_dir, batch_size, img_size):
    train_tf, val_tf = build_transforms(img_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_ds, val_ds, train_loader, val_loader


def build_model(num_classes):
    # weights = models.ResNet18_Weights.DEFAULT
    # model = models.resnet18(weights=weights)
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


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

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

    train_ds, val_ds, train_loader, val_loader = build_dataloaders(
        TRAIN_DIR, VAL_DIR, BATCH_SIZE, IMG_SIZE
    )

    num_classes = len(train_ds.classes)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print("train class_to_idx:", train_ds.class_to_idx)
    print("val class_to_idx:", val_ds.class_to_idx)
    print("same mapping?", train_ds.class_to_idx == val_ds.class_to_idx)
    print("train num classes:", len(train_ds.classes))
    print("val num classes:", len(val_ds.classes))
    print("train classes:", train_ds.classes[:10])
    print("val classes:", val_ds.classes[:10])

    # with wandb.init(
    #     project="test",
    #     name="stage2_resnet18_baseline",
    #     config={
    #         "model": "resnet18",
    #         "img_size": IMG_SIZE,
    #         "batch_size": BATCH_SIZE,
    #         "epochs": EPOCHS,
    #         "lr": LR,
    #         "num_classes": num_classes,
    #         "seed": SEED,
    #     },
    # ) as run:
    #     best_val_acc = 0.0
    #     best_state = None

    #     for epoch in range(1, EPOCHS + 1):
    #         train_loss, train_acc = train_one_epoch(
    #             model, train_loader, criterion, optimizer, device
    #         )
    #         val_loss, val_acc = validate(
    #             model, val_loader, criterion, device
    #         )

    #         print(
    #             f"[Epoch {epoch:02d}/{EPOCHS}] "
    #             f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
    #             f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    #         )

    #         wandb.log({
    #             "epoch": epoch,
    #             "train/loss": train_loss,
    #             "train/acc": train_acc,
    #             "val/loss": val_loss,
    #             "val/acc": val_acc,
    #             "lr": optimizer.param_groups[0]["lr"],
    #         })

    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             best_state = copy.deepcopy(model.state_dict())

    #     # best 저장
    #     best_path = SAVE_DIR / "best.pt"
    #     torch.save({
    #         "model_state_dict": best_state,
    #         "class_to_idx": class_to_idx,
    #         "idx_to_class": idx_to_class,
    #         "num_classes": num_classes,
    #         "img_size": IMG_SIZE,
    #     }, best_path)

    #     run.summary["best_val_acc"] = best_val_acc
    #     run.summary["best_model_path"] = str(best_path)

    #     artifact = wandb.Artifact("stage2-classifier-best", type="model")
    #     artifact.add_file(str(best_path))
    #     run.log_artifact(artifact)

    #     print(f"Best val acc: {best_val_acc:.4f}")
    #     print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    main()