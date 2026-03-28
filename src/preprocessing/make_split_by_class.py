from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


"""
    실행 순서: stage2 split 전용
    직접 실행:
    python -m src.preprocessing.make_split_by_class

    [역할]
    classification용 단일 객체 데이터셋을
    class 분포를 유지하면서 train / val로 분리하는 단계

    [전제]
    - 한 row = 한 객체(단일 이미지 또는 crop)
    - class_id 컬럼이 존재해야 함
    - stratify를 사용하므로 클래스 비율을 최대한 유지하면서 split 가능

    [하는 일]
    1. master csv 읽기
    2. class_id 기준 stratified split 수행
    3. train / val csv 저장
    4. 전체 / train / val class distribution csv 저장

    [결과]
    save_dir/
        train_annotations.csv
        val_annotations.csv
        class_distribution_all.csv
        class_distribution_train.csv
        class_distribution_val.csv
"""


MASTER_CSV = Path("data/processed/v4/master_annotations.csv")
SAVE_DIR = Path("data/processed/v4")

VAL_SIZE = 0.2
RANDOM_STATE = 42


def save_class_distribution_csv(df: pd.DataFrame, save_path: Path):
    """
    class_id별 샘플 개수와 비율을 csv로 저장하는 함수

    저장 컬럼:
    - class_id
    - count
    - ratio
    """
    dist = (
        df["class_id"]
        .value_counts()
        .sort_index()
        .rename_axis("class_id")
        .reset_index(name="count")
    )

    total = dist["count"].sum()
    dist["ratio"] = dist["count"] / total

    save_path.parent.mkdir(parents=True, exist_ok=True)
    dist.to_csv(save_path, index=False, encoding="utf-8-sig")


def make_split_by_class(
    master_csv: Path,
    save_dir: Path,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    classification용 master csv를
    class 분포를 유지하면서 train / val로 분리하는 함수

    Args:
        master_csv: 전체 단일 객체 메타데이터 csv
        save_dir: 결과 csv 저장 폴더
        val_size: validation 비율
        random_state: random seed
    """
    if not master_csv.exists():
        raise FileNotFoundError(f"master csv가 없습니다: {master_csv}")

    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_csv, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("입력 master csv가 비어 있습니다.")

    if "class_id" not in df.columns:
        raise ValueError("class_id 컬럼이 없습니다.")

    # stratify를 사용하므로
    # 최소 샘플 수가 너무 적은 클래스가 있으면 에러가 날 수 있음
    class_counts = df["class_id"].value_counts()

    # 샘플 1개짜리 class 찾기
    rare_classes = class_counts[class_counts < 2].index

    rare_df = df[df["class_id"].isin(rare_classes)]
    normal_df = df[~df["class_id"].isin(rare_classes)]

    if len(rare_df) > 0:
        print("\n[INFO] 샘플 1개짜리 클래스 발견 → val에 강제 배치")
        print(rare_df["class_id"].value_counts())

    # class 비율을 유지하면서 train / val 분리
    train_df, val_df = train_test_split(
        normal_df,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=normal_df["class_id"],
    )

    # index 정리
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # 저장 경로
    train_csv = save_dir / "train_annotations.csv"
    val_csv = save_dir / "val_annotations.csv"

    dist_all_csv = save_dir / "class_distribution_all.csv"
    dist_train_csv = save_dir / "class_distribution_train.csv"
    dist_val_csv = save_dir / "class_distribution_val.csv"

    # split csv 저장
    train_df.to_csv(train_csv, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv, index=False, encoding="utf-8-sig")

    # 분포 csv 저장
    save_class_distribution_csv(df, dist_all_csv)
    save_class_distribution_csv(train_df, dist_train_csv)
    save_class_distribution_csv(val_df, dist_val_csv)

    # 로그 출력
    print("=" * 60)
    print("make_split_by_class 완료")
    print("=" * 60)
    print(f"master_csv : {master_csv}")
    print(f"save_dir    : {save_dir}")
    print(f"total       : {len(df)}")
    print(f"train       : {len(train_df)}")
    print(f"val         : {len(val_df)}")
    print(f"train_csv   : {train_csv}")
    print(f"val_csv     : {val_csv}")
    print(f"dist_all    : {dist_all_csv}")
    print(f"dist_train  : {dist_train_csv}")
    print(f"dist_val    : {dist_val_csv}")

    print("\n[전체 class 분포]")
    print(df["class_id"].value_counts().sort_index())

    print("\n[train class 분포]")
    print(train_df["class_id"].value_counts().sort_index())

    print("\n[val class 분포]")
    print(val_df["class_id"].value_counts().sort_index())

    return {
        "train_csv": train_csv,
        "val_csv": val_csv,
        "class_distribution_all_csv": dist_all_csv,
        "class_distribution_train_csv": dist_train_csv,
        "class_distribution_val_csv": dist_val_csv,
        "num_total": len(df),
        "num_train": len(train_df),
        "num_val": len(val_df),
    }


def main():
    output = make_split_by_class(
        master_csv=MASTER_CSV,
        save_dir=SAVE_DIR,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\n분할 완료")
    print(f"train_csv: {output['train_csv']}")
    print(f"val_csv: {output['val_csv']}")


if __name__ == "__main__":
    main()