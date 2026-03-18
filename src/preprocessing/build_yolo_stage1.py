from pathlib import Path
import pandas as pd
import shutil

"""
    실행 순서: 3

    [역할]
    Stage1 (객체 탐지) 학습을 위한 YOLO 형식 데이터셋을 생성하는 단계

    [하는 일]
    1.train_annotations.csv, val_annotations.csv 읽음
    2.이미지별 bbox 묶음
    3.bbox를 YOLO 포맷 (x_center, y_center, w, h)으로 변환
    4.이미지와 label(.txt) 파일 생성(YOLO 형식 txt)
    5.원본 이미지 복사
    6.data.yaml 생성
    [결과]
    data/processed/.../yolo_stage1/
"""

TRAIN_CSV = Path("data/processed/train_annotations.csv")
VAL_CSV = Path("data/processed/val_annotations.csv")
RAW_IMG_DIR = Path("data/raw/train_images")

SAVE_ROOT = Path("data/processed/yolo_stage1")
TRAIN_IMG_OUT = SAVE_ROOT / "images" / "train"
VAL_IMG_OUT = SAVE_ROOT / "images" / "val"
TRAIN_LABEL_OUT = SAVE_ROOT / "labels" / "train"
VAL_LABEL_OUT = SAVE_ROOT / "labels" / "val"

for p in [TRAIN_IMG_OUT, VAL_IMG_OUT, TRAIN_LABEL_OUT, VAL_LABEL_OUT]:
    p.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

def convert_to_yolo(df, image_out_dir, label_out_dir, raw_img_dir):
    grouped = df.groupby("file_name")

    for file_name, group in grouped:
        img_path = raw_img_dir / file_name
        if not img_path.exists():
            print(f"[경고] 이미지 없음: {img_path}")
            continue

        # 이미지 복사
        shutil.copy2(img_path, image_out_dir / file_name)

        # 같은 이미지의 bbox들을 하나의 txt로 저장
        label_path = label_out_dir / f"{Path(file_name).stem}.txt"

        lines = []
        for _, row in group.iterrows():
            img_w = row["width"]
            img_h = row["height"]

            x = row["bbox_x"]
            y = row["bbox_y"]
            w = row["bbox_w"]
            h = row["bbox_h"]

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Stage 1은 알약 1클래스
            class_id = 0

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            lines.append(line)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

convert_to_yolo(train_df, TRAIN_IMG_OUT, TRAIN_LABEL_OUT, RAW_IMG_DIR)
convert_to_yolo(val_df, VAL_IMG_OUT, VAL_LABEL_OUT, RAW_IMG_DIR)

# data.yaml 생성
yaml_text = f"""path: {SAVE_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: pill
"""

with open(SAVE_ROOT / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_text)

print("YOLO Stage1 데이터셋 생성 완료")
print(SAVE_ROOT)

label_files = list((Path("data/processed/yolo_stage1/labels/train")).glob("*.txt"))

for lf in label_files[:3]:
    print(f"\n[{lf.name}]")
    with open(lf, "r", encoding="utf-8") as f:
        print(f.read())