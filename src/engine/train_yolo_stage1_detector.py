from pathlib import Path
import wandb
from ultralytics import YOLO

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
    └── yolo11n/
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


def upload_run_outputs(run, save_dir: Path):
    files_to_log = [
        save_dir / "results.csv",
        save_dir / "args.yaml",
        save_dir / "F1_curve.png",
        save_dir / "P_curve.png",
        save_dir / "R_curve.png",
        save_dir / "PR_curve.png",
        save_dir / "confusion_matrix.png",
        save_dir / "confusion_matrix_normalized.png",
        save_dir / "weights" / "best.pt",
        save_dir / "weights" / "last.pt",
    ]

    for path in files_to_log:
        if path.exists():
            run.save(str(path), policy="now")

    artifact = wandb.Artifact("stage1-detector-v1-yolo11n", type="model")

    best_model = save_dir / "weights" / "best.pt"
    last_model = save_dir / "weights" / "last.pt"

    if best_model.exists():
        artifact.add_file(str(best_model))
    if last_model.exists():
        artifact.add_file(str(last_model))

    if len(artifact.manifest.entries) > 0:
        run.log_artifact(artifact)


def train_yolo_stage1_detector(
    data_yaml: Path = DATA_YAML,
    model_name: str = "yolo11n.pt",
    epochs: int = 20,
    imgsz: int = 960,
    batch: int = 8,
    seed: int = SEED,
    workers: int = 4,
    patience: int = 15,
    project_dir: Path = Path("checkpoints/v1/stage1_detector"),
    run_name: str = "yolo11n",
):
    set_seed(seed)
    device = get_device()
    yolo_device = to_yolo_device(device)

    project_dir.mkdir(parents=True, exist_ok=True)

    with wandb.init(
        project="test",
        name=f"stage1_{run_name}_v1",
        job_type="train",
        config={
            "stage": "yolo11n_stage1_detector_v1",
            "model": model_name,
            "data": str(data_yaml),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "seed": seed,
            "workers": workers,
            "patience": patience,
            "device": yolo_device,
            "save_dir": str(project_dir / run_name),
        },
    ) as run:
        model = YOLO(model_name)

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=yolo_device,
            seed=seed,
            workers=workers,
            project=str(Path(project_dir).resolve()),
            name=run_name,
            pretrained=True,
            patience=patience,
            save=True,
            verbose=True,
            exist_ok=True,
            plots=True,
        )

        save_dir = Path(results.save_dir)
        print(f"\nsave_dir: {save_dir}")

        upload_run_outputs(run, save_dir)

        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"

        run.summary["save_dir"] = str(save_dir)
        run.summary["best_model_path"] = str(best_path)
        run.summary["last_model_path"] = str(last_path)

        return {
            "results": results,
            "save_dir": save_dir,
            "best_path": best_path,
            "last_path": last_path,
        }


def main():
    output = train_yolo_stage1_detector(
        data_yaml=DATA_YAML,
        model_name="yolo11n.pt",
        epochs=20,
        imgsz=960,
        batch=8,
        seed=SEED,
        workers=4,
        patience=15,
        project_dir=Path("checkpoints/v1/stage1_detector"),
        run_name="yolo11n",
    )

    print("\n학습 완료")
    print(f"best.pt: {output['best_path']}")
    print(f"last.pt: {output['last_path']}")


if __name__ == "__main__":
    main()