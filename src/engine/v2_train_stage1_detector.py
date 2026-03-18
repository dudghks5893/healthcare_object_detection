from pathlib import Path
import wandb
from ultralytics import YOLO

from src.utils import get_device
from src.utils import set_seed

"""
    실행 방법:
    python -m src.engine.v2_train_stage1_detector
"""

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
    ]

    for path in files_to_log:
        if path.exists():
            run.save(str(path), policy="now")

    best_model = save_dir / "weights" / "best.pt"
    if best_model.exists():
        artifact = wandb.Artifact("stage1-detector-v2-best", type="model")
        artifact.add_file(str(best_model))
        run.log_artifact(artifact)


def main():
    seed = 42
    model_name = "yolo11n.pt"
    data_yaml = "data/processed/v2/yolo_stage1/data.yaml"
    epochs = 20
    imgsz = 960
    batch = 8

    set_seed(seed)
    device = get_device()
    yolo_device = to_yolo_device(device)

    with wandb.init(
        project="test",
        name="stage1_yolo_v2_baseline",
        job_type="train",
        config={
            "stage": "stage1_detector_v2",
            "model": model_name,
            "data": data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "seed": seed,
            "device": yolo_device,
        },
    ) as run:
        model = YOLO(model_name)

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=yolo_device,
            seed=seed,
            pretrained=True,
            patience=15,
            save=True,
            verbose=True,
            project="outputs/stage1_detector_v2",
            name="baseline",
            exist_ok=True,
            plots=True,
        )

        save_dir = Path(results.save_dir)
        print(f"save_dir: {save_dir}")

        upload_run_outputs(run, save_dir)

        best_path = save_dir / "weights" / "best.pt"
        run.summary["save_dir"] = str(save_dir)
        run.summary["best_model_path"] = str(best_path)


if __name__ == "__main__":
    main()