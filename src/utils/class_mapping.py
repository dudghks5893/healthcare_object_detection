from pathlib import Path
import json
import pandas as pd


def build_class_mapping(csv_path: Path, sort_numeric: bool = True):
    """
        CSV 기반 class_to_idx / idx_to_class 생성

        Args:
            csv_path (Path): class_id 컬럼이 포함된 CSV 경로
            sort_numeric (bool): class_id를 숫자 기준으로 정렬할지 여부

        Returns:
            class_to_idx (dict): {"1900": 0, "2483": 1, ...}
            idx_to_class (dict): {0: "1900", 1: "2483", ...}
    """
    df = pd.read_csv(csv_path)

    if "class_id" not in df.columns:
        raise ValueError(f"'class_id' 컬럼이 없습니다: {csv_path}")

    class_ids = df["class_id"].astype(str).unique()

    if sort_numeric:
        class_ids = sorted(class_ids, key=lambda x: int(x))
    else:
        class_ids = sorted(class_ids)

    class_to_idx = {cls_id: idx for idx, cls_id in enumerate(class_ids)}
    idx_to_class = {idx: cls_id for cls_id, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class


def save_class_mapping_json(class_to_idx: dict, idx_to_class: dict, save_dir: Path):
    """
        class_to_idx / idx_to_class를 json 파일로 저장

        저장 결과:
        - class_to_idx.json
        - idx_to_class.json
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx_path = save_dir / "class_to_idx.json"
    idx_to_class_path = save_dir / "idx_to_class.json"

    with open(class_to_idx_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    with open(idx_to_class_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    return class_to_idx_path, idx_to_class_path

def load_class_mapping_json(json_path: Path):
    """
        저장된 class mapping json 파일 로드

        Args:
            json_path (Path): class_to_idx.json 또는 idx_to_class.json 경로

        Returns:
            dict
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data