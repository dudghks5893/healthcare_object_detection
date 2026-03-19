import json
from pathlib import Path
import pandas as pd

"""
    실행 순서: 1
    직접 실행: python -m src.preprocessing.build_master_table

    [역할]
    COCO 형식의 annotation(json)을 읽어서 학습에 사용할 master CSV를 생성하는 단계

    [하는 일]
    - images / annotations / categories 정보 파싱
    - bbox, class_id, file_name 등 필요한 정보 정리
    - 하나의 통합 CSV (master_annotations.csv) 생성

    [결과]
    data/processed/.../master_annotations.csv
"""

def build_master_table(
    annot_root: Path,
    train_img_dir: Path,
    save_path: Path,
) -> pd.DataFrame:
    json_paths = list(annot_root.rglob("*.json"))
    train_image_paths = list(train_img_dir.rglob("*.png"))

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
            bbox = ann["bbox"]  # [x, y, w, h]

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

    if df.empty:
        print("\n[경고] 생성된 DataFrame이 비어 있습니다.")
        print(f"- annotation 경로: {annot_root}")
        print(f"- image 경로: {train_img_dir}")
        return df

    print("\n==== train 이미지 파일명과 일치 여부 ====")
    sample_name = df["file_name"].iloc[0]
    sample_path = train_img_dir / sample_name
    print(f"샘플 Json file_name: {sample_name}")
    print(f"이미지 폴더에 존재 여부: {sample_path.exists()}")

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

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {save_path}")
    return df


def main():
    annot_root = Path("data/raw/v1/train_annotations")
    train_img_dir = Path("data/raw/v1/train_images")
    save_path = Path("data/processed/v1/master_annotations.csv")

    build_master_table(
        annot_root=annot_root,
        train_img_dir=train_img_dir,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()