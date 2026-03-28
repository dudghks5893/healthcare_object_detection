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
"""


def clamp(value, low, high):
    """
    value를 [low, high] 범위 안으로 제한한다.
    """
    return max(low, min(value, high))


def normalize_image_extension(file_name: str, default_ext: str = ".png") -> str:
    """
    파일명의 확장자를 정규화한다.
    지원: .jpg, .jpeg, .png
    그 외 확장자이거나 확장자가 없으면 default_ext를 사용한다.
    """
    ext = Path(file_name).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        return ext
    return default_ext


def get_pil_save_format(ext: str) -> str:
    """
    확장자에 맞는 PIL 저장 포맷명을 반환한다.
    """
    ext = ext.lower()
    if ext in [".jpg", ".jpeg"]:
        return "JPEG"
    if ext == ".png":
        return "PNG"
    raise ValueError(f"지원하지 않는 확장자입니다: {ext}")


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

    Returns
    -------
    cropped : Image.Image | None
        crop된 이미지. 유효하지 않은 crop이면 None
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

    # clamp 후에도 유효한 crop 영역이 아니면 skip
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    cropped = img.crop((x1, y1, x2, y2))
    return cropped, (x1, y1, x2, y2)


def save_cropped_image(cropped: Image.Image, save_path: Path) -> None:
    """
    확장자에 맞게 crop 이미지를 저장한다.
    jpg/jpeg/png 모두 지원한다.
    """
    ext = save_path.suffix.lower()
    save_format = get_pil_save_format(ext)

    if save_format == "JPEG":
        # JPEG는 alpha 지원 안 하므로 RGB로 보정
        if cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        cropped.save(save_path, format="JPEG", quality=95)
    elif save_format == "PNG":
        cropped.save(save_path, format="PNG", compress_level=3)


def save_fulltrain_crop_dataset(
    df: pd.DataFrame,
    raw_img_dir: Path,
    save_root: Path,
    margin_ratio: float = 0.10,
    save_ext: str | None = None,
) -> pd.DataFrame:
    """
    fulltrain용 crop dataset을 생성하여 저장한다.

    Parameters
    ----------
    save_ext : str | None
        저장 확장자 강제 지정.
        예: ".png", ".jpg", ".jpeg"
        None이면 원본 파일 확장자를 따라감.
    """
    train_dir = save_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_images = []
    invalid_bbox_count = 0
    saved_count = 0

    if save_ext is not None:
        save_ext = save_ext.lower()
        if save_ext not in [".jpg", ".jpeg", ".png"]:
            raise ValueError(f"save_ext는 .jpg, .jpeg, .png 중 하나여야 합니다. 현재값: {save_ext}")

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

        with Image.open(src_img_path) as img:
            img = img.convert("RGB")

            cropped, crop_box = crop_with_margin(
                img=img,
                x=x,
                y=y,
                w=w,
                h=h,
                margin_ratio=margin_ratio,
            )

            if cropped is None:
                invalid_bbox_count += 1
                print(
                    f"[skip] invalid crop "
                    f"| file={file_name} "
                    f"| ann_id={row['ann_id']} "
                    f"| bbox=({x}, {y}, {w}, {h}) "
                    f"| crop_box={crop_box}"
                )
                continue

            output_ext = save_ext if save_ext is not None else normalize_image_extension(file_name, default_ext=".png")
            crop_file_name = f"{Path(file_name).stem}_ann{int(row['ann_id'])}{output_ext}"
            crop_path = train_dir / crop_file_name

            save_cropped_image(cropped, crop_path)

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
    save_ext: str | None = None,
) -> None:
    """
    v2 master_annotations.csv를 기반으로
    Stage2 classifier용 full-train crop dataset을 생성한다.

    Parameters
    ----------
    save_ext : str | None
        crop 저장 확장자.
        None이면 원본 확장자를 따라감.
        예: ".png", ".jpg", ".jpeg"
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
    print(f"save_ext: {save_ext if save_ext is not None else '원본 확장자 유지'}")

    crop_df = save_fulltrain_crop_dataset(
        df=df,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
        margin_ratio=margin_ratio,
        save_ext=save_ext,
    )

    metadata_dir = save_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv_path = metadata_dir / "train_crop_labels.csv"
    crop_df.to_csv(metadata_csv_path, index=False, encoding="utf-8-sig")

    print("\n==== 저장 완료 ====")
    print(f"crop metadata: {metadata_csv_path}")

    print("\n==== 최종 검증 ====")
    print(f"crop 수: {len(crop_df)}")

    if crop_df.empty:
        print("class 수: 0")
        print("[경고] 저장된 crop이 없습니다.")
        return

    print(f"class 수: {crop_df['class_id'].nunique()}")

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
        save_ext=None,   # None이면 원본 확장자 유지
        # save_ext=".png",   # 전부 png로 저장하고 싶으면 이렇게
        # save_ext=".jpg",   # 전부 jpg로 저장하고 싶으면 이렇게
    )


if __name__ == "__main__":
    main()