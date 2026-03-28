import json
from pathlib import Path


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def folder_name_to_class_id(folder_name: str) -> int:
    """
    예:
        K-010221 -> 10221
        K-023203 -> 23203
    """
    if not folder_name.startswith("K-"):
        raise ValueError(f"폴더명이 'K-'로 시작하지 않습니다: {folder_name}")

    raw_number = folder_name.replace("K-", "").strip()

    if not raw_number.isdigit():
        raise ValueError(f"'K-' 뒤가 숫자가 아닙니다: {folder_name}")

    return int(raw_number)  # 앞자리 0 자동 제거


def build_image_id_map(train_img_dir: Path) -> dict[str, int]:
    """
    train_images 폴더의 모든 이미지 파일에 대해
    basename 기준 고유 image_id를 생성한다.

    반환 예:
        {
            "abc": 1,
            "def": 2,
        }
    """
    if not train_img_dir.exists():
        raise FileNotFoundError(f"train_images 폴더가 없습니다: {train_img_dir}")

    image_paths = []
    for ext in VALID_IMAGE_EXTS:
        image_paths.extend(train_img_dir.rglob(f"*{ext}"))
        image_paths.extend(train_img_dir.rglob(f"*{ext.upper()}"))

    # basename 중복 체크
    stem_to_paths = {}
    for img_path in image_paths:
        stem = img_path.stem
        stem_to_paths.setdefault(stem, []).append(img_path)

    duplicated = {stem: paths for stem, paths in stem_to_paths.items() if len(paths) > 1}
    if duplicated:
        examples = list(duplicated.items())[:10]
        msg_lines = ["basename이 중복되는 이미지가 있습니다. basename 기준 매핑이 불가능합니다:"]
        for stem, paths in examples:
            msg_lines.append(f"- {stem}: {[str(p) for p in paths]}")
        raise ValueError("\n".join(msg_lines))

    sorted_stems = sorted(stem_to_paths.keys())

    image_id_map = {}
    for idx, stem in enumerate(sorted_stems, start=1):
        image_id_map[stem] = idx

    print(f"train image 수: {len(sorted_stems)}")
    return image_id_map


def find_json_files_in_object_folder(object_folder: Path) -> list[Path]:
    """
    K-xxxxxx 폴더 안의 json 파일들을 찾는다.
    1~3개 정도 있다고 했으므로 여러 개를 허용한다.
    """
    json_files = sorted(object_folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"json 파일이 없습니다: {object_folder}")

    return json_files


def find_matching_image_id(
    data: dict,
    json_path: Path,
    image_id_map: dict[str, int],
) -> int:
    """
    json이 참조하는 이미지의 image_id를 찾는다.

    우선순위:
    1) images[0]["file_name"]의 stem
    2) json 파일명 stem

    예:
        abc.json -> abc.png
        또는 images[].file_name == abc.png
    """
    candidate_stems = []

    images = data.get("images", [])
    if images:
        file_name = images[0].get("file_name")
        if file_name:
            candidate_stems.append(Path(file_name).stem)

    candidate_stems.append(json_path.stem)

    for stem in candidate_stems:
        if stem in image_id_map:
            return image_id_map[stem]

    raise KeyError(
        f"매칭되는 이미지 파일을 찾지 못했습니다. "
        f"json={json_path}, candidates={candidate_stems}"
    )


def update_one_json_file(
    json_path: Path,
    class_id: int,
    image_id_map: dict[str, int],
    next_annotation_id: int,
    backup: bool = False,
) -> int:
    """
    단일 json 파일을 수정한다.

    수정 대상:
    - annotations[].category_id
    - categories[].id
    - categories[].name
    - images[].id
    - annotations[].image_id
    - annotations[].id (전역 고유값으로 재부여)

    Returns
    -------
    int
        다음 annotation 시작 id
    """
    if backup:
        backup_path = json_path.with_suffix(".bak.json")
        if not backup_path.exists():
            backup_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = find_matching_image_id(data, json_path, image_id_map)

    # images 수정
    for img in data.get("images", []):
        img["id"] = image_id

    # annotations 수정
    annotations = data.get("annotations", [])
    for ann in annotations:
        ann["category_id"] = class_id
        ann["image_id"] = image_id
        ann["id"] = next_annotation_id
        next_annotation_id += 1

    # categories 수정
    for cat in data.get("categories", []):
        cat["id"] = class_id
        cat["name"] = str(class_id)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(
        f"[완료] json={json_path} | class_id={class_id} | image_id={image_id} | ann_count={len(annotations)}"
    )

    return next_annotation_id


def process_one_object_folder(
    object_folder: Path,
    image_id_map: dict[str, int],
    next_annotation_id: int,
    backup: bool = False,
) -> int:
    """
    하나의 K-xxxxxx 폴더를 처리한다.
    폴더 안의 모든 json 파일은 동일한 class_id를 사용한다.
    """
    folder_name = object_folder.name
    class_id = folder_name_to_class_id(folder_name)
    json_files = find_json_files_in_object_folder(object_folder)

    print(f"\n[객체 폴더] {object_folder}")
    print(f"class_id={class_id}, json 수={len(json_files)}")

    for json_path in json_files:
        next_annotation_id = update_one_json_file(
            json_path=json_path,
            class_id=class_id,
            image_id_map=image_id_map,
            next_annotation_id=next_annotation_id,
            backup=backup,
        )

    return next_annotation_id


def process_all_json_folders(
    processing_root: Path,
    train_img_dir: Path,
    backup: bool = False,
) -> None:
    """
    전체 processing 폴더를 순회하며 모든 json을 수정한다.
    """
    if not processing_root.exists():
        raise FileNotFoundError(f"processing 폴더가 없습니다: {processing_root}")

    image_id_map = build_image_id_map(train_img_dir)

    group_dirs = sorted(
        [p for p in processing_root.iterdir() if p.is_dir() and p.name.endswith("_json")]
    )

    if not group_dirs:
        raise ValueError(f"_json 폴더를 찾지 못했습니다: {processing_root}")

    total_group_count = 0
    total_object_folder_count = 0
    total_json_count = 0
    success_json_count = 0
    fail_count = 0

    next_annotation_id = 1

    print(f"\n_json 그룹 폴더 수: {len(group_dirs)}")

    for group_dir in group_dirs:
        total_group_count += 1
        print(f"\n========== 그룹 시작: {group_dir} ==========")

        object_folders = sorted(
            [p for p in group_dir.iterdir() if p.is_dir() and p.name.startswith("K-")]
        )

        print(f"객체 폴더 수: {len(object_folders)}")

        for object_folder in object_folders:
            total_object_folder_count += 1

            json_files = list(object_folder.glob("*.json"))
            total_json_count += len(json_files)

            try:
                next_annotation_id = process_one_object_folder(
                    object_folder=object_folder,
                    image_id_map=image_id_map,
                    next_annotation_id=next_annotation_id,
                    backup=backup,
                )
                success_json_count += len(json_files)
            except Exception as e:
                fail_count += 1
                print(f"[실패] folder={object_folder} | error={e}")

    print("\n========== 전체 작업 완료 ==========")
    print(f"그룹 폴더 수: {total_group_count}")
    print(f"객체 폴더 수: {total_object_folder_count}")
    print(f"json 파일 수: {total_json_count}")
    print(f"성공 json 수: {success_json_count}")
    print(f"실패 폴더 수: {fail_count}")
    print(f"마지막 annotation_id: {next_annotation_id - 1}")


def main():
    # 정상 작동 테스트 ex)주소
    # processing_root = Path("data/raw/v4/processing")
    # train_img_dir = Path("data/raw/v4/processing_images")

    processing_root = Path("data/raw/v4/train_annotations")
    train_img_dir = Path("data/raw/v4/train_images")

    process_all_json_folders(
        processing_root=processing_root,
        train_img_dir=train_img_dir,
        backup=False,   # 원본 백업 원하면 True
    )


if __name__ == "__main__":
    main()