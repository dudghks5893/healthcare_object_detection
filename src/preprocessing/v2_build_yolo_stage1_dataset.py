import shutil
from pathlib import Path

import pandas as pd
import yaml


"""
    실행 순서: v2-3
    직접 실행: python -m src.preprocessing.v2_build_yolo_stage1_dataset

    [역할]
    v2 train/val annotation csv를 2-stage 전략용 YOLO Stage1 dataset 형태로 변환

    [2-stage 전략]
    - Stage1 (YOLO): 알약 위치만 탐지 → single-class detection
    - Stage2 (Classifier): crop 후 실제 class_id / class_name 분류

    [하는 일]
    - train / val 이미지 복사
    - YOLO 형식 label txt 생성
    - 모든 bbox label의 class를 0(pill)로 저장
    - data.yaml 생성
    - 기본 검증 출력

    [결과]
    data/processed/v2/yolo_stage1_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
"""


def make_yolo_label_line(x, y, w, h, img_w, img_h):
    """
    single-class detection이므로 class_idx=0 고정
    YOLO format: class x_center y_center w h
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    class_idx = 0
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def save_yolo_split(
    df: pd.DataFrame,
    split_name: str,
    raw_img_dir: Path,
    save_root: Path,
):
    image_save_dir = save_root / "images" / split_name
    label_save_dir = save_root / "labels" / split_name

    image_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("file_name")

    missing_images = []
    total_images = 0
    total_labels = 0

    for file_name, group in grouped:
        src_img_path = raw_img_dir / file_name

        if not src_img_path.exists():
            missing_images.append(str(src_img_path))
            continue

        # 이미지 복사
        dst_img_path = image_save_dir / file_name
        shutil.copy2(src_img_path, dst_img_path)

        # label txt 저장
        label_path = label_save_dir / f"{Path(file_name).stem}.txt"
        lines = []

        for _, row in group.iterrows():
            line = make_yolo_label_line(
                x=float(row["bbox_x"]),
                y=float(row["bbox_y"]),
                w=float(row["bbox_w"]),
                h=float(row["bbox_h"]),
                img_w=float(row["width"]),
                img_h=float(row["height"]),
            )
            lines.append(line)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        total_images += 1
        total_labels += len(lines)

    print(f"\n==== {split_name} 저장 결과 ====")
    print(f"이미지 수: {total_images}")
    print(f"라벨 수(bbox 수): {total_labels}")
    print(f"누락 이미지 수: {len(missing_images)}")

    if missing_images:
        print("[경고] 일부 이미지가 없어 복사되지 않았습니다.")
        for p in missing_images[:10]:
            print(p)
        if len(missing_images) > 10:
            print(f"... 외 {len(missing_images) - 10}개")


def build_v2_yolo_stage1_dataset(
    train_csv: Path,
    val_csv: Path,
    raw_img_dir: Path,
    save_root: Path,
):
    if not train_csv.exists():
        raise FileNotFoundError(f"train csv가 없습니다: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val csv가 없습니다: {val_csv}")
    if not raw_img_dir.exists():
        raise FileNotFoundError(f"raw image 폴더가 없습니다: {raw_img_dir}")

    train_df = pd.read_csv(train_csv, encoding="utf-8-sig")
    val_df = pd.read_csv(val_csv, encoding="utf-8-sig")

    if train_df.empty:
        raise ValueError("train csv가 비어 있습니다.")
    if val_df.empty:
        raise ValueError("val csv가 비어 있습니다.")

    required_cols = [
        "file_name",
        "width",
        "height",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    for df_name, df in [("train", train_df), ("val", val_df)]:
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} csv에 필수 컬럼이 없습니다: {missing_cols}")

    print("\n==== 2-stage Stage1 YOLO dataset 생성 ====")
    print("모든 bbox label class는 0 (pill)로 저장됩니다.")

    print("\n==== 입력 데이터 정보 ====")
    print(f"Train 이미지 수: {train_df['file_name'].nunique()}")
    print(f"Val 이미지 수: {val_df['file_name'].nunique()}")
    print(f"Train bbox 수: {len(train_df)}")
    print(f"Val bbox 수: {len(val_df)}")

    # split별 저장
    save_yolo_split(
        df=train_df,
        split_name="train",
        raw_img_dir=raw_img_dir,
        save_root=save_root,
    )

    save_yolo_split(
        df=val_df,
        split_name="val",
        raw_img_dir=raw_img_dir,
        save_root=save_root,
    )

    # single-class detector용 data.yaml 생성
    data_yaml = {
        "path": str(save_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pill"],
    }

    data_yaml_path = save_root / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data_yaml,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print("\n==== 저장 완료 ====")
    print(f"YOLO dataset root: {save_root}")
    print(f"data.yaml: {data_yaml_path}")

    # 최종 검증
    train_img_count = len(list((save_root / "images/train").glob("*")))
    val_img_count = len(list((save_root / "images/val").glob("*")))
    train_label_count = len(list((save_root / "labels/train").glob("*.txt")))
    val_label_count = len(list((save_root / "labels/val").glob("*.txt")))

    print("\n==== 최종 검증 ====")
    print(f"train images: {train_img_count}")
    print(f"val images: {val_img_count}")
    print(f"train labels: {train_label_count}")
    print(f"val labels: {val_label_count}")

    # txt 샘플 하나 확인
    sample_train_labels = list((save_root / "labels/train").glob("*.txt"))
    if sample_train_labels:
        sample_label_path = sample_train_labels[0]
        print("\n==== 샘플 라벨 파일 확인 ====")
        print(f"sample label path: {sample_label_path}")
        with open(sample_label_path, "r", encoding="utf-8") as f:
            sample_lines = f.read().splitlines()[:5]
        for line in sample_lines:
            print(line)


def main():
    train_csv = Path("data/processed/v2/train_annotations.csv")
    val_csv = Path("data/processed/v2/val_annotations.csv")
    raw_img_dir = Path("data/raw/v2/train_images")
    save_root = Path("data/processed/v2/yolo_stage1_dataset")

    build_v2_yolo_stage1_dataset(
        train_csv=train_csv,
        val_csv=val_csv,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
    )


if __name__ == "__main__":
    main()