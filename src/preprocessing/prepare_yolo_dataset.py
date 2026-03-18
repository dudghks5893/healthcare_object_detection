import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from PIL import Image


# =========================================
# 1. 경로 설정 (🔥 여기 중요)
# =========================================
BASE_DIR = Path(r"C:\Users\jgi01\OneDrive\바탕 화면\Codit_ML\healthcare_object_detection")  # sprint_ai_project1_data가 들어있는 데이터 경로 설정

RAW_DATA_DIR = BASE_DIR / "sprint_ai_project1_data"
TRAIN_IMG_DIR = RAW_DATA_DIR / "train_images"
TRAIN_ANN_DIR = RAW_DATA_DIR / "train_annotations"

YOLO_DIR = BASE_DIR / "yolo_dataset"

IMG_TRAIN_DIR = YOLO_DIR / "images" / "train"
IMG_VAL_DIR = YOLO_DIR / "images" / "val"
LBL_TRAIN_DIR = YOLO_DIR / "labels" / "train"
LBL_VAL_DIR = YOLO_DIR / "labels" / "val"

for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================================
# 2. JSON 파싱
# =========================================
def load_annotations():
    image_to_objects = defaultdict(list)
    category_ids = set()

    json_paths = list(TRAIN_ANN_DIR.rglob("*.json"))
    print(f"총 JSON 개수: {len(json_paths)}")

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            img_info = data["images"][0]
            ann = data["annotations"][0]
            cat = data["categories"][0]
        except:
            continue

        file_name = img_info["file_name"]
        bbox = ann["bbox"]
        category_id = ann["category_id"]

        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        image_to_objects[file_name].append({
            "bbox": bbox,
            "category_id": category_id
        })

        category_ids.add(category_id)

    return image_to_objects, sorted(category_ids)


# =========================================
# 3. class mapping
# =========================================
def make_class_mapping(category_ids):
    return {cat_id: idx for idx, cat_id in enumerate(category_ids)}


# =========================================
# 4. YOLO label 저장
# =========================================
def save_yolo_label(label_path, objects, img_w, img_h, class_to_idx):
    with open(label_path, "w") as f:
        for obj in objects:
            x, y, w, h = obj["bbox"]
            cls = class_to_idx[obj["category_id"]]

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w /= img_w
            h /= img_h

            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


# =========================================
# 5. 데이터 저장
# =========================================
def export_split(file_list, image_to_objects, class_to_idx, img_dir, lbl_dir):

    for file_name in file_list:
        src_img = TRAIN_IMG_DIR / file_name

        if not src_img.exists():
            print(f"[경고] 이미지 없음: {file_name}")
            continue

        dst_img = img_dir / file_name
        dst_lbl = lbl_dir / file_name.replace(".png", ".txt")

        shutil.copy2(src_img, dst_img)

        with Image.open(src_img) as img:
            w, h = img.size

        save_yolo_label(
            dst_lbl,
            image_to_objects[file_name],
            w,
            h,
            class_to_idx
        )


# =========================================
# 6. data.yaml 생성
# =========================================
def save_yaml(class_to_idx):
    yaml_path = YOLO_DIR / "data.yaml"

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DIR.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")

        for i in range(len(idx_to_class)):
            f.write(f"  {i}: '{idx_to_class[i]}'\n")

    print("data.yaml 생성 완료")


# =========================================
# 7. main
# =========================================
def main():
    image_to_objects, category_ids = load_annotations()

    print(f"이미지 수: {len(image_to_objects)}")
    print(f"클래스 수: {len(category_ids)}")

    class_to_idx = make_class_mapping(category_ids)

    all_files = list(image_to_objects.keys())

    train_files, val_files = train_test_split(
        all_files,
        test_size=0.2,
        random_state=42
    )

    print(f"train: {len(train_files)}, val: {len(val_files)}")

    export_split(train_files, image_to_objects, class_to_idx, IMG_TRAIN_DIR, LBL_TRAIN_DIR)
    export_split(val_files, image_to_objects, class_to_idx, IMG_VAL_DIR, LBL_VAL_DIR)

    save_yaml(class_to_idx)

    print("✅ YOLO 데이터셋 준비 완료")


if __name__ == "__main__":
    main()
    