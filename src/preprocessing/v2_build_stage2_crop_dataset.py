from pathlib import Path

import pandas as pd
from PIL import Image


"""
    실행 순서: v2-4
    직접 실행: python -m src.preprocessing.v2_build_stage2_crop_dataset

    [역할]
    v2 train/val annotation csv를 이용해 Stage2 classifier용 crop dataset 생성

    [하는 일]
    - bbox 기준으로 알약 crop 이미지 생성
    - margin_ratio 만큼 bbox 주변 여백 추가
    - train / val crop 이미지 저장
    - crop metadata csv 저장

    [결과]
    data/processed/v2/stage2_classifier_crop_dataset/
    ├── train/
    ├── val/
    └── metadata/
        ├── train_crop_labels.csv
        └── val_crop_labels.csv
"""


def clamp(value, low, high):
    return max(low, min(value, high))


def crop_with_margin(img: Image.Image, x, y, w, h, margin_ratio=0.10):
    img_w, img_h = img.size

    margin_w = w * margin_ratio
    margin_h = h * margin_ratio

    x1 = clamp(int(x - margin_w), 0, img_w)
    y1 = clamp(int(y - margin_h), 0, img_h)
    x2 = clamp(int(x + w + margin_w), 0, img_w)
    y2 = clamp(int(y + h + margin_h), 0, img_h)

    cropped = img.crop((x1, y1, x2, y2))
    return cropped, (x1, y1, x2, y2)


def save_split_crop_dataset(
    df: pd.DataFrame,
    split_name: str,
    raw_img_dir: Path,
    save_root: Path,
    margin_ratio: float = 0.10,
):
    split_dir = save_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_images = []
    saved_count = 0

    for _, row in df.iterrows():
        file_name = row["file_name"]
        src_img_path = raw_img_dir / file_name

        if not src_img_path.exists():
            missing_images.append(str(src_img_path))
            continue

        img = Image.open(src_img_path).convert("RGB")

        cropped, crop_box = crop_with_margin(
            img=img,
            x=float(row["bbox_x"]),
            y=float(row["bbox_y"]),
            w=float(row["bbox_w"]),
            h=float(row["bbox_h"]),
            margin_ratio=margin_ratio,
        )

        crop_file_name = f"{Path(file_name).stem}_ann{int(row['ann_id'])}.jpg"
        crop_path = split_dir / crop_file_name
        cropped.save(crop_path, quality=95)

        x1, y1, x2, y2 = crop_box

        rows.append({
            "source_file_name": file_name,
            "crop_file_name": crop_file_name,
            "crop_path": str(crop_path),
            "image_id": row["image_id"],
            "ann_id": row["ann_id"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "bbox_x": row["bbox_x"],
            "bbox_y": row["bbox_y"],
            "bbox_w": row["bbox_w"],
            "bbox_h": row["bbox_h"],
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "margin_ratio": margin_ratio,
            "split": split_name,
        })

        saved_count += 1

    crop_df = pd.DataFrame(rows)

    print(f"\n==== {split_name} crop 저장 결과 ====")
    print(f"저장 crop 수: {saved_count}")
    print(f"누락 이미지 수: {len(missing_images)}")

    if missing_images:
        print("[경고] 일부 이미지가 없어 crop 생성이 안 되었습니다.")
        for p in missing_images[:10]:
            print(p)
        if len(missing_images) > 10:
            print(f"... 외 {len(missing_images) - 10}개")

    return crop_df


def build_v2_stage2_crop_dataset(
    train_csv: Path,
    val_csv: Path,
    raw_img_dir: Path,
    save_root: Path,
    margin_ratio: float = 0.10,
):
    if not train_csv.exists():
        raise FileNotFoundError(f"train csv가 없습니다: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"val csv가 없습니다: {val_csv}")
    if not raw_img_dir.exists():
        raise FileNotFoundError(f"raw image 폴더가 없습니다: {raw_img_dir}")

    train_df = pd.read_csv(train_csv, encoding="utf-8-sig")
    val_df = pd.read_csv(val_csv, encoding="utf-8-sig")

    required_cols = [
        "file_name",
        "image_id",
        "ann_id",
        "class_id",
        "class_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]

    for df_name, df in [("train", train_df), ("val", val_df)]:
        if df.empty:
            raise ValueError(f"{df_name} csv가 비어 있습니다.")
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{df_name} csv에 필수 컬럼이 없습니다: {missing_cols}")

    print("\n==== v2 Stage2 crop dataset 생성 시작 ====")
    print(f"Train bbox 수: {len(train_df)}")
    print(f"Val bbox 수: {len(val_df)}")
    print(f"margin_ratio: {margin_ratio}")

    train_crop_df = save_split_crop_dataset(
        df=train_df,
        split_name="train",
        raw_img_dir=raw_img_dir,
        save_root=save_root,
        margin_ratio=margin_ratio,
    )

    val_crop_df = save_split_crop_dataset(
        df=val_df,
        split_name="val",
        raw_img_dir=raw_img_dir,
        save_root=save_root,
        margin_ratio=margin_ratio,
    )

    metadata_dir = save_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = metadata_dir / "train_crop_labels.csv"
    val_csv_path = metadata_dir / "val_crop_labels.csv"

    train_crop_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    val_crop_df.to_csv(val_csv_path, index=False, encoding="utf-8-sig")

    print("\n==== 저장 완료 ====")
    print(f"train crop metadata: {train_csv_path}")
    print(f"val crop metadata: {val_csv_path}")

    print("\n==== 최종 검증 ====")
    print(f"train crop 수: {len(train_crop_df)}")
    print(f"val crop 수: {len(val_crop_df)}")
    print(f"train class 수: {train_crop_df['class_id'].nunique()}")
    print(f"val class 수: {val_crop_df['class_id'].nunique()}")


def main():
    train_csv = Path("data/processed/v2/train_annotations.csv")
    val_csv = Path("data/processed/v2/val_annotations.csv")
    raw_img_dir = Path("data/raw/v2/train_images")
    save_root = Path("data/processed/v2/stage2_classifier_crop_dataset")

    build_v2_stage2_crop_dataset(
        train_csv=train_csv,
        val_csv=val_csv,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
        margin_ratio=0.10,
    )


if __name__ == "__main__":
    main()