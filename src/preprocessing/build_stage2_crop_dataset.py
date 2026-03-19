from pathlib import Path
import pandas as pd
from PIL import Image

"""
    실행 순서: 5
    
    직접 실행: python -m src.preprocessing.build_stage2_crop_dataset

    [역할]
    Stage2 (분류 모델)을 위한 crop 이미지 데이터를 생성하는 단계

    [하는 일]
    - annotation bbox 기준으로 이미지 crop
    - 각 crop 이미지에 대한 class label 매핑
    - classifier 학습용 metadata CSV 생성

    [결과]
    data/processed/v1/stage2_classifier_crop_dataset/
    crop 이미지 + train_crop_labels.csv / val_crop_labels.csv
"""

TRAIN_CSV = Path("data/processed/v1/train_annotations.csv")
VAL_CSV = Path("data/processed/v1/val_annotations.csv")
RAW_IMG_DIR = Path("data/raw/v1/train_images")

SAVE_ROOT = Path("data/processed/v1/stage2_classifier_crop_dataset")
TRAIN_OUT = SAVE_ROOT / "train"
VAL_OUT = SAVE_ROOT / "val"
META_OUT = SAVE_ROOT / "metadata"

MARGIN_RATIO = 0.10  # 10% margin


def make_crop_dataset(
    df: pd.DataFrame,
    split_name: str,
    out_dir: Path,
    raw_img_dir: Path,
    margin_ratio: float = 0.10,
):
    rows = []

    grouped = df.groupby("file_name")

    for file_name, group in grouped:
        img_path = raw_img_dir / file_name
        if not img_path.exists():
            print(f"[경고] 이미지 없음: {img_path}")
            continue

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        for _, row in group.iterrows():
            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            w = float(row["bbox_w"])
            h = float(row["bbox_h"])

            class_id = int(row["class_id"])
            class_name = str(row["class_name"])
            ann_id = row["ann_id"] if "ann_id" in row else "na"

            # margin 적용
            mx = w * margin_ratio
            my = h * margin_ratio

            x1 = max(0, int(x - mx))
            y1 = max(0, int(y - my))
            x2 = min(img_w, int(x + w + mx))
            y2 = min(img_h, int(y + h + my))

            if x2 <= x1 or y2 <= y1:
                print(f"[스킵] 잘못된 crop: {file_name}, ann_id={ann_id}")
                continue

            crop = image.crop((x1, y1, x2, y2))

            class_dir = out_dir / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)

            crop_name = f"{Path(file_name).stem}__ann_{ann_id}.png"
            crop_path = class_dir / crop_name
            crop.save(crop_path)

            rows.append({
                "split": split_name,
                "crop_path": str(crop_path),
                "crop_file": crop_name,
                "file_name": file_name,
                "class_id": class_id,
                "class_name": class_name,
                "ann_id": ann_id,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "margin_ratio": margin_ratio,
            })

    return pd.DataFrame(rows)


def build_stage2_crop_dataset(
    train_csv: Path,
    val_csv: Path,
    raw_img_dir: Path,
    save_root: Path,
    margin_ratio: float = 0.10,
    show_samples: bool = False,
    num_sample_images: int = 5,
):
    train_out = save_root / "train"
    val_out = save_root / "val"
    meta_out = save_root / "metadata"

    meta_out.mkdir(parents=True, exist_ok=True)
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if train_df.empty or val_df.empty:
        print("\n[경고] train 또는 val CSV가 비어 있습니다.")
        print(f"- train_csv: {train_csv}")
        print(f"- val_csv: {val_csv}")
        return None

    train_meta = make_crop_dataset(
        train_df,
        split_name="train",
        out_dir=train_out,
        raw_img_dir=raw_img_dir,
        margin_ratio=margin_ratio,
    )
    val_meta = make_crop_dataset(
        val_df,
        split_name="val",
        out_dir=val_out,
        raw_img_dir=raw_img_dir,
        margin_ratio=margin_ratio,
    )

    train_meta_path = meta_out / "train_crop_labels.csv"
    val_meta_path = meta_out / "val_crop_labels.csv"

    train_meta.to_csv(train_meta_path, index=False, encoding="utf-8-sig")
    val_meta.to_csv(val_meta_path, index=False, encoding="utf-8-sig")

    print("=== Stage2 crop dataset 생성 완료 ===")
    print(f"Train crop 수: {len(train_meta)}")
    print(f"Val crop 수: {len(val_meta)}")
    print(f"Train class 수: {train_meta['class_id'].nunique()}")
    print(f"Val class 수: {val_meta['class_id'].nunique()}")
    print(f"Train metadata 저장: {train_meta_path}")
    print(f"Val metadata 저장: {val_meta_path}")

    if show_samples:
        sample_paths = list((save_root / "train").rglob("*.png"))[:num_sample_images]
        print("\n==== 샘플 crop 확인 ====")
        for p in sample_paths:
            img = Image.open(p)
            print(p, img.size)
            img.show()

    return {
        "train_meta": train_meta,
        "val_meta": val_meta,
        "train_meta_path": train_meta_path,
        "val_meta_path": val_meta_path,
        "train_out": train_out,
        "val_out": val_out,
        "meta_out": meta_out,
    }


def main():
    build_stage2_crop_dataset(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        raw_img_dir=RAW_IMG_DIR,
        save_root=SAVE_ROOT,
        margin_ratio=MARGIN_RATIO,
        show_samples=False,   # 자동화 시 False 권장
        num_sample_images=5,
    )


if __name__ == "__main__":
    main()