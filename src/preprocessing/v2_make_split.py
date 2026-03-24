from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split


"""
    실행 순서: v2-2
    직접 실행: python -m src.preprocessing.v2_make_split

    [역할]
    v2 master_annotations.csv를 이미지 단위로 train / val 분리

    [개선 사항]
    - 같은 이미지는 train/val에 동시에 들어가지 않음
    - val에 각 class가 최소 1개 이상 들어가도록 보정
    - 클래스 분포 확인 로직 포함

    [결과]
    data/processed/v2/train_annotations.csv
    data/processed/v2/val_annotations.csv
    data/processed/v2/train_class_distribution.csv
    data/processed/v2/val_class_distribution.csv
"""


def print_class_distribution(df: pd.DataFrame, name: str):
    print(f"\n==== {name} 클래스 분포(class_id, class_name) ====")

    class_dist = (
        df.groupby(["class_id", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
        .reset_index(drop=True)
    )

    class_dist["ratio"] = class_dist["count"] / class_dist["count"].sum()

    print(class_dist)
    print(f"\n{name} 총 객체 수: {len(df)}")
    print(f"{name} 총 이미지 수: {df['file_name'].nunique()}")
    print(f"{name} 총 클래스 수: {df['class_id'].nunique()}")

    return class_dist


def build_image_class_map(df: pd.DataFrame):
    """
    image -> set(classes)
    class -> set(images)
    """
    image_to_classes = (
        df.groupby("file_name")["class_id"]
        .apply(lambda x: set(x.tolist()))
        .to_dict()
    )

    class_to_images = defaultdict(set)
    for file_name, classes in image_to_classes.items():
        for class_id in classes:
            class_to_images[class_id].add(file_name)

    return image_to_classes, class_to_images


def rebalance_val_min_one_class(
    df: pd.DataFrame,
    train_images: set,
    val_images: set,
):
    """
    val에 없는 클래스를 train에서 val로 이동시켜 최소 1개 이상 포함되도록 보정
    """
    image_to_classes, class_to_images = build_image_class_map(df)

    all_classes = set(df["class_id"].unique())

    def get_present_classes(image_set):
        present = set()
        for img in image_set:
            present.update(image_to_classes[img])
        return present

    val_classes = get_present_classes(val_images)
    missing_classes = all_classes - val_classes

    print("\n==== val 보정 전 누락 클래스 ====")
    print(sorted(missing_classes))
    print(f"누락 클래스 수: {len(missing_classes)}")

    moved_images = []

    # 누락 클래스가 없어질 때까지 반복
    while missing_classes:
        best_img = None
        best_gain = 0
        best_new_classes = set()

        # 누락 클래스들을 가장 많이 커버하는 train 이미지 선택
        for img in train_images:
            classes_in_img = image_to_classes[img]
            covered = classes_in_img & missing_classes
            gain = len(covered)

            if gain > best_gain:
                best_gain = gain
                best_img = img
                best_new_classes = covered

        if best_img is None or best_gain == 0:
            print("\n[경고] 더 이상 누락 클래스를 보정할 수 없습니다.")
            break

        train_images.remove(best_img)
        val_images.add(best_img)
        moved_images.append((best_img, sorted(best_new_classes)))

        val_classes.update(image_to_classes[best_img])
        missing_classes = all_classes - val_classes

    print("\n==== val 보정으로 이동한 이미지 ====")
    print(f"이동 이미지 수: {len(moved_images)}")
    for img, covered_classes in moved_images[:20]:
        print(f"{img} -> 추가된 class_id: {covered_classes}")
    if len(moved_images) > 20:
        print(f"... 외 {len(moved_images) - 20}개")

    final_val_classes = get_present_classes(val_images)
    final_missing = all_classes - final_val_classes

    print("\n==== val 보정 후 누락 클래스 ====")
    print(sorted(final_missing))
    print(f"누락 클래스 수: {len(final_missing)}")

    return train_images, val_images, moved_images


def make_v2_split(
    master_csv: Path,
    save_dir: Path,
    val_size: float = 0.2,
    random_state: int = 42,
):
    if not master_csv.exists():
        raise FileNotFoundError(f"master csv가 없습니다: {master_csv}")

    df = pd.read_csv(master_csv, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("master csv가 비어 있습니다.")

    required_cols = ["file_name", "class_id", "class_name"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    print("\n==== master 기본 정보 ====")
    print(f"총 객체 수: {len(df)}")
    print(f"총 이미지 수: {df['file_name'].nunique()}")
    print(f"총 클래스 수: {df['class_id'].nunique()}")

    image_df = df[["file_name"]].drop_duplicates().reset_index(drop=True)

    print("\n==== split 대상 이미지 수 ====")
    print(len(image_df))

    train_image_list, val_image_list = train_test_split(
        image_df["file_name"],
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    train_images = set(train_image_list.tolist())
    val_images = set(val_image_list.tolist())

    print("\n==== 초기 split 결과 ====")
    print(f"Train 이미지 수: {len(train_images)}")
    print(f"Val 이미지 수: {len(val_images)}")

    # val 최소 1클래스 보정
    train_images, val_images, moved_images = rebalance_val_min_one_class(
        df=df,
        train_images=train_images,
        val_images=val_images,
    )

    # row 복원
    train_df = df[df["file_name"].isin(train_images)].copy()
    val_df = df[df["file_name"].isin(val_images)].copy()

    print("\n==== 최종 split 결과 ====")
    print(f"Train 이미지 수: {train_df['file_name'].nunique()}")
    print(f"Val 이미지 수: {val_df['file_name'].nunique()}")
    print(f"Train 객체 수: {len(train_df)}")
    print(f"Val 객체 수: {len(val_df)}")

    overlap = set(train_df["file_name"].unique()) & set(val_df["file_name"].unique())
    print(f"\nTrain / Val 이미지 중복 수: {len(overlap)}")
    if len(overlap) > 0:
        print("[경고] train / val에 중복 이미지가 있습니다.")
        print(list(sorted(overlap))[:10])

    # 클래스 분포 출력
    train_class_dist = print_class_distribution(train_df, "Train")
    val_class_dist = print_class_distribution(val_df, "Val")

    # val에 각 클래스 최소 1개 있는지 최종 확인
    all_classes = set(df["class_id"].unique())
    val_classes = set(val_df["class_id"].unique())
    still_missing = all_classes - val_classes

    print("\n==== 최종 val 최소 1개 클래스 확인 ====")
    print(f"전체 클래스 수: {len(all_classes)}")
    print(f"Val 클래스 수: {len(val_classes)}")
    print(f"누락 클래스 수: {len(still_missing)}")
    if still_missing:
        print("[경고] 여전히 val에 없는 클래스:")
        print(sorted(still_missing))
    else:
        print("모든 클래스가 val에 최소 1개 이상 포함되었습니다.")

    # 저장
    save_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = save_dir / "train_annotations.csv"
    val_csv_path = save_dir / "val_annotations.csv"
    move_log_path = save_dir / "val_rebalance_move_log.csv"

    train_df.to_csv(train_csv_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_csv_path, index=False, encoding="utf-8-sig")
    train_class_dist.to_csv(
        save_dir / "train_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    val_class_dist.to_csv(
        save_dir / "val_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    move_log_df = pd.DataFrame(
        [
            {
                "file_name": file_name,
                "covered_class_ids": ",".join(map(str, covered_classes)),
            }
            for file_name, covered_classes in moved_images
        ]
    )
    move_log_df.to_csv(move_log_path, index=False, encoding="utf-8-sig")

    print(f"\n저장 완료: {train_csv_path}")
    print(f"저장 완료: {val_csv_path}")
    print(f"저장 완료: {save_dir / 'train_class_distribution.csv'}")
    print(f"저장 완료: {save_dir / 'val_class_distribution.csv'}")
    print(f"저장 완료: {move_log_path}")

    return train_df, val_df


def main():
    master_csv = Path("data/processed/v2/master_annotations.csv")
    save_dir = Path("data/processed/v2")

    make_v2_split(
        master_csv=master_csv,
        save_dir=save_dir,
        val_size=0.2,
        random_state=42,
    )


if __name__ == "__main__":
    main()