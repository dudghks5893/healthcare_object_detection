import json
from pathlib import Path

import pandas as pd


"""
    실행 순서: v2-1
    직접 실행: python -m src.preprocessing.v2_build_master_table

    [역할]
    v2 COCO 형식 annotation(json)을 읽어서 학습용 master CSV를 생성

    [v1과 다른 점]
    - v2는 하나의 COCO json 안에 images / annotations / categories가 모두 들어 있음
    - 실제 학습 이미지 파일은 .jpg
    - categories[*].id 를 기준으로 annotations[*].category_id 와 매핑

    [하는 일]
    - COCO images / annotations / categories 파싱
    - bbox, class_id, class_name, file_name 등 필요한 컬럼 정리
    - 하나의 통합 CSV(master_annotations.csv) 생성
    - 클래스 분포 및 기본 이상치 확인

    [결과]
    data/processed/v2/master_annotations.csv
    data/processed/v2/class_distribution_v2.csv
"""


def build_v2_master_table(
    json_path: Path,
    train_img_dir: Path,
    save_path: Path,
) -> pd.DataFrame:
    
    if not json_path.exists():
        raise FileNotFoundError(f"annotation json이 없습니다: {json_path}")

    if not train_img_dir.exists():
        raise FileNotFoundError(f"train image 폴더가 없습니다: {train_img_dir}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print("\n==== v2 json 기본 개수 ====")
    print(f"총 images 수: {len(images)}")
    print(f"총 annotations 수: {len(annotations)}")
    print(f"총 categories 수: {len(categories)}")

    if len(images) == 0 or len(annotations) == 0 or len(categories) == 0:
        print("\n[경고] images / annotations / categories 중 비어 있는 항목이 있습니다.")
        return pd.DataFrame()

    # image_id -> image info
    # v2는 실제 학습 이미지가 .jpg이므로 img["file_name"]을 기준으로 사용
    # extra.name 은 참고용 원본 이름으로만 저장
    image_map = {}
    for img in images:
        image_id = img.get("id")
        extra = img.get("extra", {}) or {}

        image_map[image_id] = {
            "image_id": image_id,
            "file_name": img.get("file_name"),               # 실제 학습용 jpg 이름
            "original_file_name": extra.get("name"),         # 참고용 원본 png 이름
            "width": img.get("width"),
            "height": img.get("height"),
        }

    # category_id -> class_name
    cat_map = {}
    for cat in categories:
        cat_id = cat.get("id")
        cat_name = cat.get("name", "unknown")
        cat_map[cat_id] = cat_name

    rows = []
    missing_image_ids = set()
    missing_category_ids = set()

    for ann in annotations:
        image_id = ann.get("image_id")
        category_id = ann.get("category_id")

        if image_id not in image_map:
            missing_image_ids.add(image_id)
            continue

        if category_id not in cat_map:
            missing_category_ids.add(category_id)

        img_info = image_map[image_id]
        bbox = ann.get("bbox", [None, None, None, None])

        row = {
            "json_path": str(json_path),

            # 이미지 정보
            "image_id": image_id,
            "file_name": img_info["file_name"],                   # jpg
            "original_file_name": img_info["original_file_name"], # 원본 png 참고용
            "width": img_info["width"],
            "height": img_info["height"],

            # bbox
            "bbox_x": bbox[0],
            "bbox_y": bbox[1],
            "bbox_w": bbox[2],
            "bbox_h": bbox[3],

            # class
            "class_id": category_id,
            "class_name": cat_map.get(category_id, "unknown"),

            # annotation
            "ann_id": ann.get("id"),
            "area": ann.get("area"),
            "iscrowd": ann.get("iscrowd", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("\n[경고] 생성된 DataFrame이 비어 있습니다.")
        return df

    train_image_paths = list(train_img_dir.rglob("*"))
    train_image_names = {p.name for p in train_image_paths if p.is_file()}

    print("\n==== 이미지 파일명 존재 여부 샘플 확인 ====")
    sample_name = df["file_name"].iloc[0]
    sample_exists = sample_name in train_image_names
    print(f"샘플 file_name: {sample_name}")
    print(f"이미지 폴더에 존재 여부: {sample_exists}")

    print("\n==== 기본 정보 ====")
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

    print("\n==== 클래스 분포 비율 ====")
    class_dist_name = (
        df.groupby(["class_id", "class_name"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )

    class_ratio = class_dist_name.copy()
    class_ratio["ratio"] = class_ratio["count"] / class_ratio["count"].sum()
    print(class_ratio)

    if missing_image_ids:
        print("\n[경고] images에 없는 annotation.image_id가 있습니다:")
        print(sorted(missing_image_ids))

    if missing_category_ids:
        print("\n[경고] categories에 없는 annotation.category_id가 있습니다:")
        print(sorted(missing_category_ids))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    class_dist_save_path = save_path.parent / "class_distribution_v2.csv"
    class_ratio.to_csv(class_dist_save_path, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {save_path}")
    print(f"클래스 분포 저장 완료: {class_dist_save_path}")

    return df


def main():
    json_path = Path("data/raw/v2/train_annotations/_annotations.fixed.coco.json")
    train_img_dir = Path("data/raw/v2/train_images")
    save_path = Path("data/processed/v2/master_annotations.csv")

    build_v2_master_table(
        json_path=json_path,
        train_img_dir=train_img_dir,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()