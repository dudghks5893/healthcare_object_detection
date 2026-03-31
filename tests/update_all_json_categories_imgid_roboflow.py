import json
from pathlib import Path
import re

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def clean_filename(name: str) -> str:
    """
    파일명 정리:
    1) _png.rf.xxx / _jpg.rf.xxx / _jpeg.rf.xxx 제거
    2) '__' 앞부분 + '__' 자체 제거

    예:
    K-003743-012081-022627-044834__K-003743-012081-022627-044834_0_2_0_2_75_000_200.png
    ->
    K-003743-012081-022627-044834_0_2_0_2_75_000_200.png
    """
    original_name = name

    # 확장자 분리
    path_obj = Path(name)
    stem = path_obj.stem
    suffix = path_obj.suffix

    # 1. rf suffix 제거
    # 예: xxx_png.rf.abcdef -> xxx
    stem = re.sub(r"_(png|jpg|jpeg)\.rf\.[^.]+$", "", stem, flags=re.IGNORECASE)

    # 2. "__" 앞부분 제거
    # 예: A__B -> B
    if "__" in stem:
        stem = stem.split("__", 1)[1]

    cleaned = f"{stem}{suffix}"

    print(f"[DEBUG] clean_filename: {original_name} -> {cleaned}")
    return cleaned


def collect_image_paths(root_dir: Path) -> list[Path]:
    """
    이미지 파일 경로 수집
    - Windows에서 *.png, *.PNG 중복 수집될 수 있으므로 set으로 중복 제거
    """
    image_path_set = set()

    for ext in VALID_IMAGE_EXTS:
        image_path_set.update(root_dir.rglob(f"*{ext}"))
        image_path_set.update(root_dir.rglob(f"*{ext.upper()}"))

    return sorted(image_path_set)


def fix_image_filenames(root_dir: Path):
    """
    root_dir 하위 이미지 파일명을 clean_filename 규칙에 맞게 변경
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {root_dir}")

    image_paths = collect_image_paths(root_dir)

    renamed_count = 0
    skipped_count = 0

    print(f"\n총 이미지 수: {len(image_paths)}")

    for img_path in image_paths:
        original_name = img_path.name
        new_name = clean_filename(original_name)

        if original_name == new_name:
            skipped_count += 1
            continue

        new_path = img_path.with_name(new_name)

        # 충돌 방지
        if new_path.exists():
            print(f"[SKIP] 이미 존재함: {new_path}")
            skipped_count += 1
            continue

        img_path.rename(new_path)

        print(
            f"[RENAME]\n"
            f"  OLD: {original_name}\n"
            f"  NEW: {new_name}"
        )

        renamed_count += 1

    print("\n===== 파일명 변경 결과 =====")
    print(f"수정된 파일: {renamed_count}")
    print(f"건너뜀: {skipped_count}")


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
    basename(stem) 기준 고유 image_id를 생성한다.
    """
    if not train_img_dir.exists():
        raise FileNotFoundError(f"train_images 폴더가 없습니다: {train_img_dir}")

    image_paths = collect_image_paths(train_img_dir)

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
    """
    json_files = sorted(object_folder.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"json 파일이 없습니다: {object_folder}")

    return json_files


def normalize_v5_key_from_image_name(file_name: str) -> str:
    """
    이미지 파일명/JSON 내부 file_name에서
    image_id_map의 key(stem)와 맞도록 정규화한다.

    처리:
    - 공백 제거
    - _png.rf.xxx / _jpg.rf.xxx 제거
    - __ 앞부분 제거
    - 확장자 제거
    """
    name = file_name.strip()

    name = re.sub(r"_(png|jpg|jpeg)\.rf\.[^.]+$", "", name, flags=re.IGNORECASE)

    if "__" in name:
        name = name.split("__", 1)[1]

    name = Path(name).stem
    return name.strip()


def find_matching_image_id(
    data: dict,
    json_path: Path,
    image_id_map: dict[str, int],
) -> int:
    """
    json이 참조하는 이미지의 image_id를 찾는다.

    우선순위:
    1) images[0]["file_name"] 정규화 결과
    2) json 파일명 stem 정규화 결과
    """
    candidate_stems = []

    images = data.get("images", [])
    if images:
        file_name = images[0].get("file_name")
        if file_name:
            candidate_stems.append(normalize_v5_key_from_image_name(file_name))

    candidate_stems.append(normalize_v5_key_from_image_name(json_path.name))

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
    - images[].file_name
    - annotations[].category_id
    - categories[].id
    - categories[].name
    - images[].id
    - annotations[].image_id
    - annotations[].id (전역 고유값으로 재부여)
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
        old_file_name = img.get("file_name", "")
        if old_file_name:
            new_file_name = clean_filename(Path(old_file_name).name)
            img["file_name"] = new_file_name
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

    processing_root = Path("data/raw/v5/train_annotations")
    train_img_dir = Path("data/raw/v5/train_images")

    # 1. 실제 이미지 파일명 먼저 수정
    fix_image_filenames(train_img_dir)

    # 2. 수정된 파일명 기준으로 json 갱신
    process_all_json_folders(
        processing_root=processing_root,
        train_img_dir=train_img_dir,
        backup=False,   # 원본 백업 원하면 True
    )


if __name__ == "__main__":
    main()