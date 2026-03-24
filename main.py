"""
===========================================
Healthcare Object Detection Pipeline Controller
===========================================

[사용 방법]

1. 전체 파이프라인 실행
python main.py --config configs/yolo11s_resnet50_transforms_epoch_20_v2.yaml --step all

2. 전처리만 실행
python main.py --config configs/yolo11n_resnet18_v1.yaml --step preprocess

3. Stage1 (Detector 학습)
python main.py --config configs/yolo11s_resnet50_v1_test.yaml --step stage1

4. Stage2 (Classifier 학습)
python main.py --config configs/yolo11s_resnet50_transforms_epoch_20_v2.yaml --step stage2

5. 추론 (Submission 생성)
python main.py --config configs/yolo11s_resnet50_transforms_epoch_20_v2.yaml --step predict

6. 개별 단계 실행
python main.py --config configs/yolo11s_resnet50_transforms_epoch_20_v2.yaml --step 1
...
python main.py --config configs/yolo11n_resnet18_v1.yaml --step 8

-------------------------------------------
[Step 설명]

1. build_master_table              → json → CSV 변환
2. make_split                      → train/val 분리
3. build_yolo_stage1_dataset       → YOLO dataset 생성
4. train_yolo_stage1_detector      → bbox detector 학습
5. build_stage2_crop_dataset       → crop 이미지 생성
6. make_stage2_fulltrain_csv       → full-train CSV 생성
7. train_stage2_classifier         → 분류 모델 학습
8. predict_2stage                  → 최종 제출 CSV 생성

-------------------------------------------
"""

import argparse
from pathlib import Path

from src.utils import load_config

from src.preprocessing.build_master_table import build_master_table
from src.preprocessing.v2_build_master_table import build_v2_master_table
from src.preprocessing.make_split import make_split
from src.preprocessing.build_yolo_stage1_dataset import build_yolo_stage1_dataset
from src.preprocessing.v2_build_yolo_stage1_dataset import build_v2_yolo_stage1_dataset
from src.engine.train_yolo_stage1_detector import train_yolo_stage1_detector
from src.preprocessing.build_stage2_crop_dataset import build_stage2_crop_dataset
from src.preprocessing.v2_build_stage2_crop_dataset import build_v2_stage2_crop_dataset
from src.preprocessing.make_stage2_fulltrain_csv import make_stage2_fulltrain_csv
from src.engine.train_stage2_classifier import train_stage2_classifier
from src.engine.train_stage2_classifier_fulltrain import train_stage2_classifier_fulltrain
from src.engine.predict_2stage import predict_2stage


# =========================
# 경로 생성
# =========================
def build_paths(cfg):
    processed_dir = Path(cfg["paths"]["processed_dir"])
    train_img_dir = Path(cfg["paths"]["train_img_dir"])
    test_img_dir = Path(cfg["paths"]["test_img_dir"])
    annot_root = Path(cfg["paths"]["annot_root"])

    master_csv = processed_dir / "master_annotations.csv"
    train_csv = processed_dir / "train_annotations.csv"
    val_csv = processed_dir / "val_annotations.csv"

    stage1_dataset_dir = Path(cfg["stage1"]["dataset_dir"])
    stage1_checkpoint_dir = Path(cfg["stage1"]["checkpoint_dir"])

    stage2_crop_dataset_dir = Path(cfg["stage2"]["crop_dataset_dir"])
    stage2_train_csv = Path(cfg["stage2"]["train_csv"])
    stage2_val_csv = Path(cfg["stage2"]["val_csv"])
    stage2_fulltrain_csv = stage2_crop_dataset_dir / "metadata" / "full_train_crop_labels.csv"
    stage2_checkpoint_dir = Path(cfg["stage2"]["checkpoint_dir"])

    detector_weight = stage1_checkpoint_dir / cfg["stage1"]["run_name"] / "weights" / "best.pt"
    classifier_weight = stage2_checkpoint_dir / "best.pt"

    submission_csv = Path(cfg["output"]["submission_csv"])
    predict_vis_dir = Path(cfg["output"]["predict_vis_dir"])

    return {
        "annot_root": annot_root,
        "train_img_dir": train_img_dir,
        "test_img_dir": test_img_dir,
        "processed_dir": processed_dir,
        "master_csv": master_csv,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "stage1_dataset_dir": stage1_dataset_dir,
        "stage1_checkpoint_dir": stage1_checkpoint_dir,
        "stage2_crop_dataset_dir": stage2_crop_dataset_dir,
        "stage2_train_csv": stage2_train_csv,
        "stage2_val_csv": stage2_val_csv,
        "stage2_fulltrain_csv": stage2_fulltrain_csv,
        "stage2_checkpoint_dir": stage2_checkpoint_dir,
        "detector_weight": detector_weight,
        "classifier_weight": classifier_weight,
        "submission_csv": submission_csv,
        "predict_vis_dir": predict_vis_dir,
    }


# =========================
# STEP 함수들
# =========================
def step_1_build_master(cfg, paths):
    print("\n[STEP 1] build_master_table 시작")
    version = cfg["version"]

    if version == "v1":
        print("version: v1")
        build_master_table(
            annot_root=paths["annot_root"],
            train_img_dir=paths["train_img_dir"],
            save_path=paths["master_csv"],
        )
    elif version == "v2":
        print("version: v2")
        json_path = paths["annot_root"] / "_annotations.fixed.coco.json"
        build_v2_master_table(
            json_path=json_path,
            train_img_dir=paths["train_img_dir"],
            save_path=paths["master_csv"],
        )
    else:
        raise ValueError(f"지원하지 않는 version: {version}")
    
    print("[STEP 1] 완료")


def step_2_make_split(cfg, paths):
    print("\n[STEP 2] make_split 시작")
    make_split(
            master_csv=paths["master_csv"],
            save_dir=paths["processed_dir"],
            val_size=cfg["split"]["val_size"],
            random_state=cfg["split"]["random_state"],
    )
    print("[STEP 2] 완료")


def step_3_build_yolo_stage1_dataset(cfg, paths):
    print("\n[STEP 3] build_yolo_stage1_dataset 시작")
    version = cfg["version"]
    if version == "v1":
        print("version: v1")
        build_yolo_stage1_dataset(
            train_csv=paths["train_csv"],
            val_csv=paths["val_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage1_dataset_dir"],
        )
    elif version == "v2":
        print("version: v2")
        build_v2_yolo_stage1_dataset(
            train_csv=paths["train_csv"],
            val_csv=paths["val_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage1_dataset_dir"],
        )
    else:
        raise ValueError(f"지원하지 않는 version: {version}")
    
    print("[STEP 3] 완료")


def step_4_train_stage1_detector(cfg, paths):
    print("\n[STEP 4] train_yolo_stage1_detector 시작")

    stage1_cfg = cfg["stage1"]

    train_yolo_stage1_detector(
        wandb_project=cfg["stage1"]["wandb_project"],
        wandb_run_name=cfg["stage1"]["wandb_run_name"],
        data_yaml=paths["stage1_dataset_dir"] / "data.yaml",
        model_name=stage1_cfg["model_name"],
        epochs=stage1_cfg["epochs"],
        imgsz=stage1_cfg["imgsz"],
        batch=stage1_cfg["batch"],
        seed=cfg["seed"],
        workers=stage1_cfg["workers"],
        patience=stage1_cfg["patience"],
        project_dir=paths["stage1_checkpoint_dir"],
        run_name=stage1_cfg["run_name"],
        val=stage1_cfg.get("val", True),
        pretrained=stage1_cfg.get("pretrained", True),

        # optimizer / lr
        optimizer=stage1_cfg.get("optimizer"),
        lr0=stage1_cfg.get("lr0"),
        lrf=stage1_cfg.get("lrf"),
        weight_decay=stage1_cfg.get("weight_decay"),
        cos_lr=stage1_cfg.get("cos_lr"),
        warmup_epochs=stage1_cfg.get("warmup_epochs"),

        # loss weight
        box=stage1_cfg.get("box"),
        cls=stage1_cfg.get("cls"),
        dfl=stage1_cfg.get("dfl"),

        # augmentation
        hsv_h=stage1_cfg.get("hsv_h"),
        hsv_s=stage1_cfg.get("hsv_s"),
        hsv_v=stage1_cfg.get("hsv_v"),
        degrees=stage1_cfg.get("degrees"),
        translate=stage1_cfg.get("translate"),
        scale=stage1_cfg.get("scale"),
        fliplr=stage1_cfg.get("fliplr"),
        flipud=stage1_cfg.get("flipud"),
        mosaic=stage1_cfg.get("mosaic"),
        mixup=stage1_cfg.get("mixup"),
        copy_paste=stage1_cfg.get("copy_paste"),

        # 기타
        save=stage1_cfg.get("save"),
        verbose=stage1_cfg.get("verbose"),
        plots=stage1_cfg.get("plots"),
        exist_ok=stage1_cfg.get("exist_ok"),
    )
    print("[STEP 4] 완료")


def step_5_build_stage2_crop_dataset(cfg, paths):
    print("\n[STEP 5] build_stage2_crop_dataset 시작")
    version = cfg["version"]

    if version == "v1":
        print("version: v1")
        build_stage2_crop_dataset(
            train_csv=paths["train_csv"],
            val_csv=paths["val_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage2_crop_dataset_dir"],
            margin_ratio=cfg["stage2"]["crop_margin_ratio"],
            show_samples=False,
            num_sample_images=5,
        )
    elif version == "v2":
        print("version: v2")
        build_v2_stage2_crop_dataset(
            train_csv=paths["train_csv"],
            val_csv=paths["val_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage2_crop_dataset_dir"],
            margin_ratio=cfg["stage2"]["crop_margin_ratio"],
        )
    else:
        raise ValueError(f"지원하지 않는 version: {version}")
    
    print("[STEP 5] 완료")


def step_6_make_stage2_fulltrain_csv(cfg, paths):
    print("\n[STEP 6] make_stage2_fulltrain_csv 시작")
    make_stage2_fulltrain_csv(
        train_csv=paths["stage2_crop_dataset_dir"] / "metadata" / "train_crop_labels.csv",
        val_csv=paths["stage2_crop_dataset_dir"] / "metadata" / "val_crop_labels.csv",
        save_csv=paths["stage2_fulltrain_csv"],
    )
    print("[STEP 6] 완료")


def step_7_train_stage2_classifier(cfg, paths):
    print("\n[STEP 7] train_stage2_classifier 시작")

    stage2_cfg = cfg["stage2"]

    train_stage2_classifier(
        wandb_project=cfg["stage2"]["wandb_project"],
        wandb_run_name=cfg["stage2"]["wandb_run_name"],
        train_csv=paths["stage2_train_csv"],
        val_csv=paths["stage2_val_csv"],
        save_dir=paths["stage2_checkpoint_dir"],
        model_name=stage2_cfg["model_name"],
        pretrained=stage2_cfg["pretrained"],
        img_size=stage2_cfg["img_size"],
        batch_size=stage2_cfg["batch_size"],
        epochs=stage2_cfg["epochs"],
        lr=stage2_cfg["lr"],
        seed=cfg["seed"],
        num_workers=stage2_cfg["num_workers"],
        pin_memory=stage2_cfg["pin_memory"],

        # augmentation yaml 연결용
        augmentation=stage2_cfg.get("augmentation"),
        aug_preview=stage2_cfg.get("aug_preview"),
    )
    print("[STEP 7] 완료")
# def step_7_train_stage2_classifier(cfg, paths):
#     print("\n[STEP 7] train_stage2_classifier_fulltrain 시작")
#     train_stage2_classifier_fulltrain(
#         full_train_csv=paths["stage2_fulltrain_csv"],
#         save_dir=paths["stage2_checkpoint_dir"],
#         model_name=cfg["stage2"]["model_name"],
#         pretrained=cfg["stage2"]["pretrained"],
#         img_size=cfg["stage2"]["img_size"],
#         batch_size=cfg["stage2"]["batch_size"],
#         epochs=cfg["stage2"]["epochs"],
#         lr=cfg["stage2"]["lr"],
#         seed=cfg["seed"],
#         num_workers=cfg["stage2"]["num_workers"],
#         pin_memory=cfg["stage2"]["pin_memory"],
#     )
#     print("[STEP 7] 완료")


def step_8_predict(cfg, paths):
    print("\n[STEP 8] predict_2stage 시작")
    predict_2stage(
        test_img_dir=paths["test_img_dir"],
        detector_weight=paths["detector_weight"],
        classifier_weight=paths["classifier_weight"],
        classifier_dir=paths["stage2_checkpoint_dir"],
        save_csv=paths["submission_csv"],
        predict_vis_dir=paths["predict_vis_dir"],
        det_imgsz=cfg["stage1"]["imgsz"],
        det_conf=cfg["stage1"]["conf"],
        det_iou=cfg["stage1"]["iou"],
        crop_margin_ratio=cfg["stage2"]["crop_margin_ratio"],
        save_vis_limit=cfg["output"]["save_vis_limit"],
    )
    print("[STEP 8] 완료")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/yolo11n_resnet18_v1.yaml")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "preprocess", "stage1", "stage2", "predict",
                 "1", "2", "3", "4", "5", "6", "7", "8"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = build_paths(cfg)

    if args.step == "all":
        step_1_build_master(cfg, paths)
        step_2_make_split(cfg, paths)
        step_3_build_yolo_stage1_dataset(cfg, paths)
        step_4_train_stage1_detector(cfg, paths)
        step_5_build_stage2_crop_dataset(cfg, paths)
        step_6_make_stage2_fulltrain_csv(cfg, paths)
        step_7_train_stage2_classifier(cfg, paths)
        step_8_predict(cfg, paths)

    elif args.step == "preprocess":
        step_1_build_master(cfg, paths)
        step_2_make_split(cfg, paths)
        step_3_build_yolo_stage1_dataset(cfg, paths)
        step_5_build_stage2_crop_dataset(cfg, paths)
        step_6_make_stage2_fulltrain_csv(cfg, paths)

    elif args.step == "stage1":
        step_4_train_stage1_detector(cfg, paths)

    elif args.step == "stage2":
        step_7_train_stage2_classifier(cfg, paths)

    elif args.step == "predict":
        step_8_predict(cfg, paths)

    elif args.step == "1":
        step_1_build_master(cfg, paths)
    elif args.step == "2":
        step_2_make_split(cfg, paths)
    elif args.step == "3":
        step_3_build_yolo_stage1_dataset(cfg, paths)
    elif args.step == "4":
        step_4_train_stage1_detector(cfg, paths)
    elif args.step == "5":
        step_5_build_stage2_crop_dataset(cfg, paths)
    elif args.step == "6":
        step_6_make_stage2_fulltrain_csv(cfg, paths)
    elif args.step == "7":
        step_7_train_stage2_classifier(cfg, paths)
    elif args.step == "8":
        step_8_predict(cfg, paths)


if __name__ == "__main__":
    main()