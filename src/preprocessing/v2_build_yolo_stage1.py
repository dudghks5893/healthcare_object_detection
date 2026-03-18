from pathlib import Path
import pandas as pd
import shutil

TRAIN_CSV = Path("data/processed/v2/train_annotations_v2.csv")
VAL_CSV = Path("data/processed/v2/val_annotations_v2.csv")

RAW_IMG_DIR = Path("data/v2/train_images")   # 필요하면 실제 이미지 폴더명에 맞게 수정
SAVE_ROOT = Path("data/processed/v2/yolo_stage1")

TRAIN_IMG_OUT = SAVE_ROOT / "images" / "train"
VAL_IMG_OUT = SAVE_ROOT / "images" / "val"
TRAIN_LABEL_OUT = SAVE_ROOT / "labels" / "train"
VAL_LABEL_OUT = SAVE_ROOT / "labels" / "val"

for p in [TRAIN_IMG_OUT, VAL_IMG_OUT, TRAIN_LABEL_OUT, VAL_LABEL_OUT]:
    p.mkdir(parents=True, exist_ok=True)


def convert_to_yolo(df, image_out_dir, label_out_dir, raw_img_dir):
    grouped = df.groupby("file_name")

    for file_name, group in grouped:
        img_path = raw_img_dir / file_name
        if not img_path.exists():
            print(f"[경고] 이미지 없음: {img_path}")
            continue

        shutil.copy2(img_path, image_out_dir / file_name)

        label_path = label_out_dir / f"{Path(file_name).stem}.txt"
        lines = []

        for _, row in group.iterrows():
            img_w = float(row["width"])
            img_h = float(row["height"])

            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            w = float(row["bbox_w"])
            h = float(row["bbox_h"])

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            class_id = 0  # Stage1은 pill 1클래스

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            lines.append(line)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    convert_to_yolo(train_df, TRAIN_IMG_OUT, TRAIN_LABEL_OUT, RAW_IMG_DIR)
    convert_to_yolo(val_df, VAL_IMG_OUT, VAL_LABEL_OUT, RAW_IMG_DIR)

    yaml_text = f"""path: {SAVE_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: pill
"""

    with open(SAVE_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("YOLO Stage1 v2 데이터셋 생성 완료")
    print(SAVE_ROOT)

    label_files = list(Path("data/processed/v2/yolo_stage1/labels/train").glob("*.txt"))
    for lf in label_files[:3]:
        print(f"\n[{lf.name}]")
        print(lf.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()