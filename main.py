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
python main.py --config configs/yolo11s_resnet50_tr_ep_20_claasweights_deg_180_v4.yaml --step 1
...
python main.py --config configs/yolo11n_resnet18_v1.yaml --step 8

-------------------------------------------

[추론 (Submission 생성) 기준 Step 설명]
0. predict_2stage                  → 최종 제출 CSV 생성

[Split Train Data 기준 Step 설명]
1. build_master_table              → json → CSV 변환
2. make_split                      → train/val 분리
3. build_yolo_stage1_dataset       → YOLO dataset 생성
4. train_yolo_stage1_detector      → bbox detector 학습
5. build_stage2_crop_dataset       → crop 이미지 생성
6. train_stage2_classifier         → 분류 모델 학습
7. make_stage2_fulltrain_csv       → full-train CSV 생성 (Split 한거 다시 합치는 용도 필요 시 사용)



[Full Train Data 기준 Step 설명]
11. build_master_table                           → json → CSV 변환
12. v2_build_yolo_stage1_dataset_fulltrain       → YOLO Full Train Dataset 생성
13. train_yolo_stage1_detector                   → bbox detector 학습 (yaml에 stage1 부분에 val=false 설정 필요.)
14. v2_build_stage2_crop_dataset_fulltrain       → Full Train Crop Dataset 이미지 생성
15. stage2_classifier_fulltrain                  → 분류 모델 full train 학습

-------------------------------------------
"""

import argparse
from pathlib import Path

from src.utils import load_config

from src.preprocessing.build_master_table import build_master_table
from src.preprocessing.v2_build_master_table import build_v2_master_table
from src.preprocessing.make_split import make_split
from src.preprocessing.make_split_by_class import make_split_by_class
from src.preprocessing.build_yolo_stage1_dataset import build_yolo_stage1_dataset
from src.preprocessing.v2_build_yolo_stage1_dataset import build_v2_yolo_stage1_dataset
from src.preprocessing.v2_build_yolo_stage1_dataset_fulltrain import build_v2_yolo_stage1_dataset_fulltrain
from src.engine.train_yolo_stage1_detector import train_yolo_stage1_detector
from src.preprocessing.build_stage2_crop_dataset import build_stage2_crop_dataset
from src.preprocessing.v2_build_stage2_crop_dataset import build_v2_stage2_crop_dataset
from src.preprocessing.v2_build_stage2_crop_dataset_fulltrain import build_v2_stage2_crop_dataset_fulltrain
from src.preprocessing.make_stage2_fulltrain_csv import make_stage2_fulltrain_csv
from src.engine.train_stage2_classifier import train_stage2_classifier
from src.engine.train_stage2_classifier_fulltrain import train_stage2_classifier_fulltrain
from src.engine.predict_2stage import predict_2stage


# =========================
# 경로 생성
# =========================
def build_paths(cfg):
    test_img_dir = Path(cfg["paths"]["test_img_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"]) if cfg["paths"].get("processed_dir") else None
    train_img_dir = Path(cfg["paths"]["train_img_dir"]) if cfg["paths"].get("train_img_dir") else None
    annot_root = Path(cfg["paths"]["annot_root"]) if cfg["paths"].get("annot_root") else None

    stage1_processed_dir = Path(cfg["paths"]["stage1_processed_dir"]) if cfg["paths"].get("stage1_processed_dir") else None
    stage1_train_img_dir = Path(cfg["paths"]["stage1_train_img_dir"]) if cfg["paths"].get("stage1_train_img_dir") else None
    stage1_annot_root = Path(cfg["paths"]["stage1_annot_root"]) if cfg["paths"].get("stage1_annot_root") else None

    stage2_processed_dir = Path(cfg["paths"]["stage2_processed_dir"]) if cfg["paths"].get("stage2_processed_dir") else None
    stage2_train_img_dir = Path(cfg["paths"]["stage2_train_img_dir"]) if cfg["paths"].get("stage2_train_img_dir") else None
    stage2_annot_root = Path(cfg["paths"]["stage2_annot_root"]) if cfg["paths"].get("stage2_annot_root") else None

    master_csv = processed_dir / "master_annotations.csv" if processed_dir else None
    train_csv = processed_dir / "train_annotations.csv" if processed_dir else None
    val_csv = processed_dir / "val_annotations.csv" if processed_dir else None

    stage1_master_csv = (
        stage1_processed_dir / "master_annotations.csv"
        if stage1_processed_dir else None
    )
    stage2_master_csv = (
        stage2_processed_dir / "master_annotations.csv"
        if stage2_processed_dir else None
    )

    stage1_train_csv = (
        stage1_processed_dir / "train_annotations.csv"
        if stage1_processed_dir else None
    )
    stage1_val_csv = (
        stage1_processed_dir / "val_annotations.csv"
        if stage1_processed_dir else None
    )

    stage2_split_train_csv = (
        stage2_processed_dir / "train_annotations.csv"
        if stage2_processed_dir else None
    )
    stage2_split_val_csv = (
        stage2_processed_dir / "val_annotations.csv"
        if stage2_processed_dir else None
    )

    stage1_dataset_dir = Path(cfg["stage1"]["dataset_dir"])
    stage1_checkpoint_dir = Path(cfg["stage1"]["checkpoint_dir"])

    stage2_crop_dataset_dir = Path(cfg["stage2"]["crop_dataset_dir"])
    stage2_train_csv = Path(cfg["stage2"]["train_csv"]) if cfg["stage2"].get("train_csv") else None
    stage2_val_csv = Path(cfg["stage2"]["val_csv"]) if cfg["stage2"].get("val_csv") else None

    stage2_class_dist_csv = (
        processed_dir / "class_distribution_v2.csv"
        if processed_dir else None
    )

    stage2_checkpoint_dir = Path(cfg["stage2"]["checkpoint_dir"])

    detector_weight = stage1_checkpoint_dir / cfg["stage1"]["run_name"] / "weights" / "best.pt"
    classifier_weight = stage2_checkpoint_dir / "best.pt"

    submission_csv = Path(cfg["output"]["submission_csv"])
    predict_vis_dir = Path(cfg["output"]["predict_vis_dir"])

    return {
        "test_img_dir": test_img_dir,

        "annot_root": annot_root,
        "train_img_dir": train_img_dir,
        "processed_dir": processed_dir,

        "stage1_annot_root": stage1_annot_root,
        "stage1_train_img_dir": stage1_train_img_dir,
        "stage1_processed_dir": stage1_processed_dir,

        "stage2_annot_root": stage2_annot_root,
        "stage2_train_img_dir": stage2_train_img_dir,
        "stage2_processed_dir": stage2_processed_dir,

        "master_csv": master_csv,
        "train_csv": train_csv,
        "val_csv": val_csv,

        "stage1_master_csv": stage1_master_csv,
        "stage2_master_csv": stage2_master_csv,
        "stage1_train_csv": stage1_train_csv,
        "stage1_val_csv": stage1_val_csv,
        "stage2_split_train_csv": stage2_split_train_csv,
        "stage2_split_val_csv": stage2_split_val_csv,

        "stage2_class_dist_csv": stage2_class_dist_csv,

        "stage1_dataset_dir": stage1_dataset_dir,
        "stage1_checkpoint_dir": stage1_checkpoint_dir,

        "stage2_crop_dataset_dir": stage2_crop_dataset_dir,
        "stage2_train_csv": stage2_train_csv,
        "stage2_val_csv": stage2_val_csv,
        "stage2_checkpoint_dir": stage2_checkpoint_dir,

        "detector_weight": detector_weight,
        "classifier_weight": classifier_weight,

        "submission_csv": submission_csv,
        "predict_vis_dir": predict_vis_dir,
    }


# =========================
# STEP 함수들
# =========================
def step_0_predict(cfg, paths):
    print("\n[STEP 0] predict_2stage 시작")
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
    print("[STEP 0] 완료")

def step_1_build_master(cfg, paths):
    print("\n[STEP 1] build_master_table 시작")
    version = cfg["version"]

    if version == "v1" or version == "v4":
        print(f"version: {version}")
        build_master_table(
            annot_root=paths["annot_root"],
            train_img_dir=paths["train_img_dir"],
            save_path=paths["master_csv"],
        )
    else:
        print(f"version: {version}")
        annot_root = paths["annot_root"]
        if annot_root is None:
            # stage1 / stage2 데이터가 서로 다른 경우
            stage_jobs = [
                (
                    paths["stage1_annot_root"] / "_annotations.fixed.coco.json",
                    paths["stage1_train_img_dir"],
                    paths["stage1_master_csv"],
                ),
                (
                    paths["stage2_annot_root"] / "_annotations.fixed.coco.json",
                    paths["stage2_train_img_dir"],
                    paths["stage2_master_csv"],
                ),
            ]

            for json_path, train_img_dir, save_path in stage_jobs:
                build_v2_master_table(
                    json_path=json_path,
                    train_img_dir=train_img_dir,
                    save_path=save_path,
                )
        else:
            # 공통 annotation / image dir 사용하는 경우
            json_path = annot_root / "_annotations.fixed.coco.json"
            build_v2_master_table(
                json_path=json_path,
                train_img_dir=paths["train_img_dir"],
                save_path=paths["master_csv"],
            )
    print("[STEP 1] 완료")


def step_2_make_split(cfg, paths):
    print("\n[STEP 2] make_split 시작")

    annot_root = paths["annot_root"]

    if annot_root is None:
        split_jobs = [
            {
                "name": "stage1",
                "master_csv": paths["stage1_master_csv"],
                "save_dir": paths["stage1_processed_dir"],
            },
            {
                "name": "stage2",
                "master_csv": paths["stage2_master_csv"],
                "save_dir": paths["stage2_processed_dir"],
            },
        ]

        for job in split_jobs:
            print(f"[STEP 2] {job['name']} split 진행")
            make_split(
                master_csv=job["master_csv"],
                save_dir=job["save_dir"],
                val_size=cfg["split"]["val_size"],
                random_state=cfg["split"]["random_state"],
            )

    else:
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
    if version == "v1" or version == "v4":
        print(f"version: {version}")
        build_yolo_stage1_dataset(
            train_csv=paths["train_csv"],
            val_csv=paths["val_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage1_dataset_dir"],
        )
    else:
        print(f"version: {version}")
        train_csv = paths["train_csv"] or paths["stage1_train_csv"]
        val_csv = paths["val_csv"] or paths["stage1_val_csv"]
        raw_img_dir = paths["train_img_dir"] or paths["stage1_train_img_dir"]
        build_v2_yolo_stage1_dataset(
            train_csv=train_csv,
            val_csv=val_csv,
            raw_img_dir=raw_img_dir,
            save_root=paths["stage1_dataset_dir"],
        )
    
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

    if version == "v1" or version == "v4":
        print(f"version: {version}")
        build_stage2_crop_dataset(
            train_csv=paths["train_csv"],
            val_csv=paths["val_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage2_crop_dataset_dir"],
            margin_ratio=cfg["stage2"]["crop_margin_ratio"],
            show_samples=False,
            num_sample_images=5,
        )
    else:
        print(f"version: {version}")
        train_csv = paths["train_csv"] or paths["stage2_split_train_csv"]
        val_csv = paths["val_csv"] or paths["stage2_split_val_csv"]
        raw_img_dir = paths["train_img_dir"] or paths["stage2_train_img_dir"]
        build_v2_stage2_crop_dataset(
            train_csv=train_csv,
            val_csv=val_csv,
            raw_img_dir=raw_img_dir,
            save_root=paths["stage2_crop_dataset_dir"],
            margin_ratio=cfg["stage2"]["crop_margin_ratio"],
        )
    
    print("[STEP 5] 완료")

def step_6_train_stage2_classifier(cfg, paths):
    print("\n[STEP 6] train_stage2_classifier 시작")

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
    print("[STEP 6] 완료")


def step_7_make_stage2_fulltrain_csv(cfg, paths):
    print("\n[STEP 7] make_stage2_fulltrain_csv 시작")
    make_stage2_fulltrain_csv(
        train_csv=paths["stage2_crop_dataset_dir"] / "metadata" / "train_crop_labels.csv",
        val_csv=paths["stage2_crop_dataset_dir"] / "metadata" / "val_crop_labels.csv",
        save_csv=paths["stage2_fulltrain_csv"],
    )
    print("[STEP 7] 완료")


def step_12_build_yolo_stage1_full_dataset(cfg, paths):
    print("\n[STEP 12] build_yolo_stage1_dataset 시작")
    version = cfg["version"]
    if version == "v1" or version == "v4":
        print(f"version: {version}")
        build_v2_yolo_stage1_dataset_fulltrain(
            master_csv=paths["master_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage1_dataset_dir"],
        )
    else:
        print(f"version: {version}")
        build_v2_yolo_stage1_dataset_fulltrain(
            master_csv=paths["master_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage1_dataset_dir"],
        )
    
    print("[STEP 12] 완료")


def step_14_build_stage2_crop_full_dataset(cfg, paths):
    print("\n[STEP 14] build_yolo_stage1_dataset 시작")
    build_v2_stage2_crop_dataset_fulltrain(
            master_csv=paths["master_csv"],
            raw_img_dir=paths["train_img_dir"],
            save_root=paths["stage2_crop_dataset_dir"],
            margin_ratio=cfg["stage2"]["crop_margin_ratio"],
            save_ext=None,   # None이면 원본 확장자 유지
            # save_ext=".png",   # 전부 png로 저장하고 싶으면 이렇게
            # save_ext=".jpg",   # 전부 jpg로 저장하고 싶으면 이렇게
        )
    print("[STEP 14] 완료")


def step_15_train_stage2_classifier_fulltrain(cfg, paths):
    print("\n[STEP 15] train_stage2_classifier_fulltrain 시작")

    stage2_cfg = cfg["stage2"]

    train_stage2_classifier_fulltrain(
        wandb_project=stage2_cfg["wandb_project"],
        wandb_run_name=stage2_cfg["wandb_run_name"],
        fulltrain_csv=paths["stage2_train_csv"],
        class_dist_csv=paths["stage2_class_dist_csv"],
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
        augmentation=stage2_cfg.get("augmentation"),
        aug_preview=stage2_cfg.get("aug_preview"),
    )

    print("[STEP 15] 완료")

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
                 "0", "1", "2", "3", "4", "5", "6", "7", "11", "12", "13", "14", "15"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = build_paths(cfg)

    if args.step == "all":
        step_0_predict(cfg, paths)
        step_1_build_master(cfg, paths)
        step_2_make_split(cfg, paths)
        step_3_build_yolo_stage1_dataset(cfg, paths)
        step_4_train_stage1_detector(cfg, paths)
        step_5_build_stage2_crop_dataset(cfg, paths)
        step_6_train_stage2_classifier(cfg, paths)

    elif args.step == "preprocess":
        step_1_build_master(cfg, paths)
        step_2_make_split(cfg, paths)
        step_3_build_yolo_stage1_dataset(cfg, paths)
        step_5_build_stage2_crop_dataset(cfg, paths)

    elif args.step == "stage1":
        step_4_train_stage1_detector(cfg, paths)

    elif args.step == "stage2":
        step_6_train_stage2_classifier(cfg, paths)

    elif args.step == "predict":
        step_0_predict(cfg, paths)

    elif args.step == "0":
        step_0_predict(cfg, paths)
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
        step_6_train_stage2_classifier(cfg, paths)
    elif args.step == "7":
        step_7_make_stage2_fulltrain_csv(cfg, paths)(cfg, paths)
    elif args.step == "11":
        print("[Full Train]")
        step_1_build_master(cfg, paths)(cfg, paths)
    elif args.step == "12":
        print("[Full Train]")
        step_12_build_yolo_stage1_full_dataset(cfg, paths)
    elif args.step == "13":
        print("[Full Train]")
        step_4_train_stage1_detector(cfg, paths)
    elif args.step == "14":
        print("[Full Train]")
        step_14_build_stage2_crop_full_dataset(cfg, paths)
    elif args.step == "15":
        print("[Full Train]")
        step_15_train_stage2_classifier_fulltrain(cfg, paths)


if __name__ == "__main__":
    main()