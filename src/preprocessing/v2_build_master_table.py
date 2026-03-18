import json
from pathlib import Path
import pandas as pd

ANNOT_PATH = Path("data/v2/train_annotations/_annotations.coco.json")
SAVE_PATH = Path("data/processed/v2/master_annotations_v2.csv")

with open(ANNOT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images_df = pd.DataFrame(data["images"])
annotations_df = pd.DataFrame(data["annotations"])
categories_df = pd.DataFrame(data["categories"])

# image_id -> file_name, width, height
images_df = images_df.rename(columns={"id": "image_id"})

# category_id -> class_name
categories_df = categories_df.rename(columns={"id": "class_id", "name": "class_name"})

# annotations의 bbox 분리
annotations_df[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]] = pd.DataFrame(
    annotations_df["bbox"].tolist(), index=annotations_df.index
)

# merge
df = annotations_df.merge(
    images_df[["image_id", "file_name", "width", "height"]],
    on="image_id",
    how="left"
).merge(
    categories_df[["class_id", "class_name"]],
    left_on="category_id",
    right_on="class_id",
    how="left"
)

# 필요한 컬럼만 정리
df = df[
    [
        "file_name",
        "width",
        "height",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "category_id",
        "class_name",
        "image_id",
        "id",
        "area",
    ]
].copy()

df = df.rename(columns={
    "category_id": "class_id",
    "id": "ann_id",
})

print("==== 기본 정보 ====")
print(df.head())
print("총 객체 수:", len(df))
print("총 이미지 수:", df["file_name"].nunique())
print("총 클래스 수:", df["class_id"].nunique())

print("\n==== bbox 이상치 확인 ====")
print("bbox_x 음수:", (df["bbox_x"] < 0).sum())
print("bbox_y 음수:", (df["bbox_y"] < 0).sum())
print("bbox_w <= 0:", (df["bbox_w"] <= 0).sum())
print("bbox_h <= 0:", (df["bbox_h"] <= 0).sum())

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")

print(f"\n저장 완료: {SAVE_PATH}")