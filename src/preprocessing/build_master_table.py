import json
from pathlib import Path
import pandas as pd

"""
    1.train_annotations 아래의 모든 json 탐색
    2.각 json에서
        * 이미지 파일명
        * bbox
        * class id
        * class name
    추출
    3.전부 하나의 dataframe으로 합침
    4.master_annotations.csv 저장
"""

ANNOT_ROOT = Path("data/raw/train_annotations")
TRAIN_IMG_DIR = Path("data/raw/train_images")
SAVE_PATH = Path("data/processed/master_annotations.csv")

json_paths = list(ANNOT_ROOT.rglob("*.json"))
train_image_paths = list(TRAIN_IMG_DIR.rglob("*.png"))

rows = []

for json_path in json_paths:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if len(images) == 0 or len(annotations) == 0:
        continue

    img = images[0]

    # category id -> name 매핑
    cat_map = {c["id"]: c["name"] for c in categories}

    for ann in annotations:
        bbox = ann["bbox"]   # [x, y, w, h]

        row = {
            "json_path": str(json_path),
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],
            "class_id": ann["category_id"],
            "class_name": cat_map.get(ann["category_id"], "unknown"),
            "image_id": ann.get("image_id"),
            "ann_id": ann.get("id"),
            "area": ann.get("area"),
        }
        rows.append(row)

df = pd.DataFrame(rows)

print("\n==== 총 개수 ====")
print(f"총 json 개수: {len(json_paths)}")
print(f"총 Train Image 개수: {len(train_image_paths)}")

print("\n==== train 이미지 파일명과 일치 여부 ====")
sample_name = df["file_name"].iloc[0]
sample_path = TRAIN_IMG_DIR / sample_name
print(f"샘플 Json file_name: {sample_name}")
print(f"이미지 폴더에 존재 여부: {sample_path.exists()}")

print(sample_name)
print(sample_path.exists())

print("\n==== Json 기본 정보 ====")
print(f"총 행 수(객체 수): {len(df)}")
print(f"총 이미지 수: {df['file_name'].nunique()}")
print(f"총 클래스 수: {df['class_id'].nunique()}")

print("\n==== bbox 이상치 확인 ====")
print("bbox_x 음수:", (df["bbox_x"] < 0).sum())
print("bbox_y 음수:", (df["bbox_y"] < 0).sum())
print("bbox_w <= 0:", (df["bbox_w"] <= 0).sum())
print("bbox_h <= 0:", (df["bbox_h"] <= 0).sum())

print("\n==== 이미지당 객체 수 ====")
obj_per_image = df.groupby("file_name").size().describe()
print(obj_per_image)

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {SAVE_PATH}")