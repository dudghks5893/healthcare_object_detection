from pathlib import Path
from ultralytics import YOLO
import wandb
from src.utils import get_device
from src.utils import set_seed

"""
    실행 순서: 4

    직접 실행: python -m src.engine.train_yolo_stage1_detector

    [역할]
    YOLO 기반 객체 탐지 모델(Stage1 Detector)을 학습하는 단계

    [하는 일]
    - YOLO 모델 로드 및 학습 수행
    - bbox 검출 성능 학습
    - checkpoints 경로에 가중치 저장
    - W&B에 실험 로그 및 주요 산출물 업로드

    [결과]
    checkpoints/v1/stage1_detector/
    └── yolo11s/
        ├── weights/
        │   ├── best.pt
        │   └── last.pt
        ├── results.csv
        ├── args.yaml
        └── ...
"""

DATA_YAML = Path("data/processed/v1/yolo_stage1_dataset/data.yaml")
SEED = 42


def to_yolo_device(device):
    if device.type == "cuda":
        return "0"
    elif device.type == "mps":
        return "mps"
    return "cpu"

def train_yolo_stage1_detector(
    wandb_project: str = "test",
    wandb_run_name: str = "train_yolo_stage1_detector",
    data_yaml: Path = DATA_YAML,
    model_name: str = "yolo11s.pt",
    epochs: int = 20,
    imgsz: int = 768,
    batch: int = 8,
    seed: int = SEED,
    workers: int = 4,
    patience: int = 10,
    project_dir: Path = Path("checkpoints/v1/stage1_detector"),
    run_name: str = "yolo11s",
    val: bool = True,
    pretrained: bool = True,

    # optimizer / lr
    optimizer: str | None = None,
    lr0: float | None = None,
    lrf: float | None = None,
    weight_decay: float | None = None,
    cos_lr: bool | None = None,
    warmup_epochs: float | None = None,

    # loss weight
    box: float | None = None,
    cls: float | None = None,
    dfl: float | None = None,

    # augmentation
    hsv_h: float | None = None,
    hsv_s: float | None = None,
    hsv_v: float | None = None,
    degrees: float | None = None,
    translate: float | None = None,
    scale: float | None = None,
    fliplr: float | None = None,
    flipud: float | None = None,
    mosaic: float | None = None,
    mixup: float | None = None,
    copy_paste: float | None = None,

    # 기타
    save: bool | None = None,
    verbose: bool | None = None,
    plots: bool | None = None,
    exist_ok: bool | None = None,
):
    set_seed(seed)
    device = get_device()
    yolo_device = to_yolo_device(device)

    project_dir.mkdir(parents=True, exist_ok=True)

    # W&B metric logging 활성화
    """
        Ultralytics 내부에서 자동으로 wandb run 생성
        그래서 project 이름을 직접 못 넣음
        대신 환경변수로 override 해야 함
    """

    wandb.init(project=wandb_project, name=wandb_run_name, job_type="training")

    model = YOLO(model_name)
    # 필수 파라미터
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": yolo_device,
        "seed": seed,
        "workers": workers,
        "project": str(Path(project_dir).resolve()),
        "name": run_name,
        "pretrained": pretrained,
        "patience": patience,
        "val": val,
    }

    # 선택 파라미터: 설정 하고 싶은 값만 설정해서 전달 가능.
    optional_train_kwargs = {
        # optimizer / lr
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "weight_decay": weight_decay,
        "cos_lr": cos_lr,
        "warmup_epochs": warmup_epochs,

        # loss weight
        "box": box,
        "cls": cls,
        "dfl": dfl,

        # augmentation
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "fliplr": fliplr,
        "flipud": flipud,
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,

        # 기타
        "save": save,
        "verbose": verbose,
        "plots": plots,
        "exist_ok": exist_ok,
    }

    train_kwargs.update({k: v for k, v in optional_train_kwargs.items() if v is not None})

    results = model.train(**train_kwargs)

    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    last_path = save_dir / "weights" / "last.pt"

    print(f"\nsave_dir: {save_dir}")

    return {
        "results": results,
        "save_dir": save_dir,
        "best_path": best_path,
        "last_path": last_path,
    }


def main():
    output = train_yolo_stage1_detector(
        data_yaml=DATA_YAML,
        model_name="yolo11s.pt",
        epochs=20,
        imgsz=768,
        batch=8,
        seed=SEED,
        workers=4,
        patience=10,
        project_dir=Path("checkpoints/v1/stage1_detector"),
        run_name="yolo11s",
        val=True,
        pretrained=True,

        # 필요할 때만 사용
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=2.0,

        # 예시: 아래는 안 넘기면 Ultralytics 기본값 사용
        # box=7.5,
        # cls=0.5,
        # dfl=1.5,
        # hsv_h=0.0,
        # hsv_s=0.0,
        # hsv_v=0.0,
        # degrees=0.0,
        # translate=0.0,
        # scale=0.0,
        # fliplr=0.0,
        # flipud=0.0,
        # mosaic=0.0,
        # mixup=0.0,
        # copy_paste=0.0,
        # save=True,
        # verbose=True,
        # plots=True,
        # exist_ok=True,
    )

    print("\n학습 완료")
    print(f"best.pt: {output['best_path']}")
    print(f"last.pt: {output['last_path']}")


if __name__ == "__main__":
    main()