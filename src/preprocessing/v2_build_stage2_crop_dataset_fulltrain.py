from pathlib import Path

import pandas as pd
from PIL import Image


"""
실행 예시:
    python -m src.preprocessing.v2_build_stage2_crop_dataset_fulltrain

[역할]
- v2 master_annotations.csv를 기반으로
  Stage2 classifier용 full-train crop dataset을 생성하는 단계

[상황]
- 이번 v2 데이터는 train / val split 없이
  전체 annotation을 전부 train으로 사용하여 classifier를 학습할 예정
- 따라서 crop 이미지는 train/ 폴더 하나에만 저장
- metadata도 fulltrain용 csv 하나만 저장

[입력]
- data/processed/v2/master_annotations.csv
- data/raw/v2/train_images/

[출력]
- data/processed/v2/stage2_classifier_crop_dataset_fulltrain/
    ├── train/
    └── metadata/
        └── fulltrain_crop_labels.csv

[하는 일]
- bbox 기준으로 알약 crop 이미지 생성
- margin_ratio 만큼 bbox 주변 여백 추가
- crop 이미지 저장
- crop metadata csv 저장
"""


def clamp(value, low, high):
    """
    value를 [low, high] 범위 안으로 제한한다.
    """
    return max(low, min(value, high))


def crop_with_margin(
    img: Image.Image,
    x: float,
    y: float,
    w: float,
    h: float,
    margin_ratio: float = 0.10,
):
    """
    bbox 주변에 margin을 추가하여 crop한다.

    Parameters
    ----------
    img : Image.Image
        원본 이미지
    x, y, w, h : float
        COCO bbox 형식 (x_min, y_min, width, height)
    margin_ratio : float
        bbox 크기 대비 여백 비율

    Returns
    -------
    cropped : Image.Image
        crop된 이미지
    crop_box : tuple[int, int, int, int]
        실제 crop에 사용된 좌표 (x1, y1, x2, y2)
    """
    img_w, img_h = img.size

    margin_w = w * margin_ratio
    margin_h = h * margin_ratio

    x1 = clamp(int(x - margin_w), 0, img_w)
    y1 = clamp(int(y - margin_h), 0, img_h)
    x2 = clamp(int(x + w + margin_w), 0, img_w)
    y2 = clamp(int(y + h + margin_h), 0, img_h)

    cropped = img.crop((x1, y1, x2, y2))
    return cropped, (x1, y1, x2, y2)


def save_fulltrain_crop_dataset(
    df: pd.DataFrame,
    raw_img_dir: Path,
    save_root: Path,
    margin_ratio: float = 0.10,
) -> pd.DataFrame:
    """
    fulltrain용 crop dataset을 생성하여 저장한다.

    생성되는 구조:
        save_root/
            └── train/

    Returns
    -------
    pd.DataFrame
        crop metadata dataframe
    """
    train_dir = save_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_images = []
    invalid_bbox_count = 0
    saved_count = 0

    for _, row in df.iterrows():
        file_name = row["file_name"]
        src_img_path = raw_img_dir / file_name

        if not src_img_path.exists():
            missing_images.append(str(src_img_path))
            continue

        x = float(row["bbox_x"])
        y = float(row["bbox_y"])
        w = float(row["bbox_w"])
        h = float(row["bbox_h"])

        # 비정상 bbox는 제외
        if w <= 0 or h <= 0:
            invalid_bbox_count += 1
            continue

        img = Image.open(src_img_path).convert("RGB")

        cropped, crop_box = crop_with_margin(
            img=img,
            x=x,
            y=y,
            w=w,
            h=h,
            margin_ratio=margin_ratio,
        )

        crop_file_name = f"{Path(file_name).stem}_ann{int(row['ann_id'])}.jpg"
        crop_path = train_dir / crop_file_name
        cropped.save(crop_path, quality=95)

        x1, y1, x2, y2 = crop_box

        rows.append(
            {
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
                "split": "train",
            }
        )

        saved_count += 1

    crop_df = pd.DataFrame(rows)

    print("\n==== fulltrain crop 저장 결과 ====")
    print(f"저장 crop 수: {saved_count}")
    print(f"누락 이미지 수: {len(missing_images)}")
    print(f"제외된 비정상 bbox 수: {invalid_bbox_count}")

    if missing_images:
        print("\n[경고] 일부 이미지가 없어 crop 생성이 안 되었습니다.")
        for p in missing_images[:10]:
            print(p)
        if len(missing_images) > 10:
            print(f"... 외 {len(missing_images) - 10}개")

    return crop_df


def build_v2_stage2_crop_dataset_fulltrain(
    master_csv: Path,
    raw_img_dir: Path,
    save_root: Path,
    margin_ratio: float = 0.10,
) -> None:
    """
    v2 master_annotations.csv를 기반으로
    Stage2 classifier용 full-train crop dataset을 생성한다.

    Parameters
    ----------
    master_csv : Path
        v2 master_annotations.csv 경로
    raw_img_dir : Path
        원본 이미지 폴더 경로
    save_root : Path
        crop dataset 저장 루트 경로
    margin_ratio : float
        bbox 주변 여백 비율
    """
    if not master_csv.exists():
        raise FileNotFoundError(f"master csv가 없습니다: {master_csv}")
    if not raw_img_dir.exists():
        raise FileNotFoundError(f"raw image 폴더가 없습니다: {raw_img_dir}")

    df = pd.read_csv(master_csv, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("master csv가 비어 있습니다.")

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
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"master csv에 필수 컬럼이 없습니다: {missing_cols}")

    numeric_cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_len = len(df)
    df = df.dropna(subset=numeric_cols).copy()
    dropped_len = before_len - len(df)

    print("\n==== v2 Stage2 fulltrain crop dataset 생성 시작 ====")
    print(f"전체 annotation 수: {before_len}")
    print(f"NaN 제거 후 annotation 수: {len(df)}")
    print(f"제거된 annotation 수: {dropped_len}")
    print(f"전체 이미지 수: {df['file_name'].nunique()}")
    print(f"전체 class 수: {df['class_id'].nunique()}")
    print(f"margin_ratio: {margin_ratio}")

    crop_df = save_fulltrain_crop_dataset(
        df=df,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
        margin_ratio=margin_ratio,
    )

    metadata_dir = save_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv_path = metadata_dir / "fulltrain_crop_labels.csv"
    crop_df.to_csv(metadata_csv_path, index=False, encoding="utf-8-sig")

    print("\n==== 저장 완료 ====")
    print(f"crop metadata: {metadata_csv_path}")

    print("\n==== 최종 검증 ====")
    print(f"crop 수: {len(crop_df)}")
    print(f"class 수: {crop_df['class_id'].nunique()}")

    if not crop_df.empty:
        print("\n==== 샘플 metadata ====")
        print(crop_df.head(3).to_string(index=False))


def main():
    master_csv = Path("data/processed/v2/master_annotations.csv")
    raw_img_dir = Path("data/raw/v2/train_images")
    save_root = Path("data/processed/v2/stage2_classifier_crop_dataset_fulltrain")

    build_v2_stage2_crop_dataset_fulltrain(
        master_csv=master_csv,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
        margin_ratio=0.10,
    )


if __name__ == "__main__":
    main()