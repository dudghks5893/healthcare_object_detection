import json
from pathlib import Path


"""
[역할]
v2 COCO JSON에서

1) categories[*].id 를 categories[*].name 의 숫자값으로 변경
2) annotations[*].category_id 도 위 변경에 맞게 함께 수정

[입력 예시]
categories:
    {"id": 25, "name": "2483"}
annotations:
    {"category_id": 25}

[출력 예시]
categories:
    {"id": 2483, "name": "2483"}
annotations:
    {"category_id": 2483}

[실행]
python -m tests.fix_v2_coco_category_ids
"""


INPUT_JSON = Path("data/raw/v2/train_annotations/_annotations.coco.json")
OUTPUT_JSON = Path("data/raw/v2/train_annotations/_annotations.fixed.coco.json")


def fix_category_ids(input_json: Path, output_json: Path):
    if not input_json.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])
    annotations = data.get("annotations", [])

    if not categories:
        raise ValueError("categories가 비어 있습니다.")

    # old_id -> new_id(name의 숫자값) 매핑
    id_map = {}

    new_categories = []
    seen_new_ids = set()

    for cat in categories:
        old_id = cat["id"]
        name = str(cat["name"]).strip()

        # "pill" 같은 상위 category는 그대로 둘지, 제외할지 선택
        # 여기서는 숫자 name만 변환 대상으로 사용
        if name.isdigit():
            new_id = int(name)
        else:
            # 숫자가 아닌 카테고리는 그대로 유지
            new_id = old_id

        id_map[old_id] = new_id

        new_cat = cat.copy()
        new_cat["id"] = new_id

        # 중복 id 방지
        if new_id not in seen_new_ids:
            new_categories.append(new_cat)
            seen_new_ids.add(new_id)

    # annotations category_id 수정
    new_annotations = []
    missing_old_ids = set()

    for ann in annotations:
        old_cat_id = ann["category_id"]

        if old_cat_id not in id_map:
            missing_old_ids.add(old_cat_id)
            new_ann = ann.copy()
        else:
            new_ann = ann.copy()
            new_ann["category_id"] = id_map[old_cat_id]

        new_annotations.append(new_ann)

    # 결과 반영
    data["categories"] = new_categories
    data["annotations"] = new_annotations

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n[완료] category id 수정 완료")
    print(f"입력 파일: {input_json}")
    print(f"출력 파일: {output_json}")

    print("\n[매핑 예시 10개]")
    for i, (old_id, new_id) in enumerate(id_map.items()):
        print(f"{old_id} -> {new_id}")
        if i >= 9:
            break

    if missing_old_ids:
        print("\n[경고] categories에 없는 annotation category_id가 있습니다:")
        print(sorted(missing_old_ids))
    else:
        print("\n모든 annotations.category_id가 정상적으로 매핑되었습니다.")


def main():
    fix_category_ids(INPUT_JSON, OUTPUT_JSON)


if __name__ == "__main__":
    main()