from pathlib import Path
import pandas as pd
from PIL import Image

"""
    실행 방법:
    터미널에 python -m src.preprocessing.build_stage2_crops 입력
"""

TRAIN_CSV = Path("data/processed/train_annotations.csv")
VAL_CSV = Path("data/processed/val_annotations.csv")
RAW_IMG_DIR = Path("data/raw/train_images")

SAVE_ROOT = Path("data/processed/stage2_classifier")
TRAIN_OUT = SAVE_ROOT / "train"
VAL_OUT = SAVE_ROOT / "val"
META_OUT = SAVE_ROOT / "metadata"

MARGIN_RATIO = 0.10  # 10% margin


def make_crop_dataset(df: pd.DataFrame, split_name: str, out_dir: Path):
    rows = []

    grouped = df.groupby("file_name")

    for file_name, group in grouped:
        img_path = RAW_IMG_DIR / file_name
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
            mx = w * MARGIN_RATIO
            my = h * MARGIN_RATIO

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
                "margin_ratio": MARGIN_RATIO,
            })

    return pd.DataFrame(rows)


def main():
    META_OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    VAL_OUT.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_meta = make_crop_dataset(train_df, "train", TRAIN_OUT)
    val_meta = make_crop_dataset(val_df, "val", VAL_OUT)

    train_meta.to_csv(META_OUT / "train_crop_labels.csv", index=False, encoding="utf-8-sig")
    val_meta.to_csv(META_OUT / "val_crop_labels.csv", index=False, encoding="utf-8-sig")

    print("=== Stage2 crop dataset 생성 완료 ===")
    print(f"Train crop 수: {len(train_meta)}")
    print(f"Val crop 수: {len(val_meta)}")
    print(f"Train class 수: {train_meta['class_id'].nunique()}")
    print(f"Val class 수: {val_meta['class_id'].nunique()}")

    sample_paths = list(Path("data/processed/stage2_classifier/train").rglob("*.png"))[:5]

    for p in sample_paths:
        img = Image.open(p)
        print(p, img.size)
        img.show()


if __name__ == "__main__":
    main()