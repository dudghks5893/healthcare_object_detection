from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    실행 순서: 2
    직접 실행: python -m src.preprocessing.make_split

    [역할]
    master CSV를 기반으로 train / validation 데이터를 분리하는 단계

    [하는 일]
    - 이미지 단위로 데이터 분할 (데이터 누수 방지)
    - train_annotations.csv / val_annotations.csv 생성

    [결과]
    data/processed/.../train_annotations.csv
    data/processed/.../val_annotations.csv
"""

def make_split(
    master_csv: Path,
    save_dir: Path,
    val_size: float = 0.2,
    random_state: int = 42,
):
    df = pd.read_csv(master_csv)

    print("==== master 정보 ====")
    print(f"총 객체 수: {len(df)}")
    print(f"총 이미지 수: {df['file_name'].nunique()}")
    print(f"총 클래스 수: {df['class_id'].nunique()}")

    if df.empty:
        print("\n[경고] master CSV가 비어 있습니다.")
        print(f"- master_csv: {master_csv}")
        return None

    # 이미지 단위 split
    unique_images = df["file_name"].drop_duplicates().values

    train_imgs, val_imgs = train_test_split(
        unique_images,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )

    train_df = df[df["file_name"].isin(train_imgs)].reset_index(drop=True)
    val_df = df[df["file_name"].isin(val_imgs)].reset_index(drop=True)

    train_img_df = pd.DataFrame({"file_name": sorted(train_imgs)})
    val_img_df = pd.DataFrame({"file_name": sorted(val_imgs)})

    print("\n==== split 결과 ====")
    print(f"Train 이미지 수: {train_df['file_name'].nunique()}")
    print(f"Val 이미지 수: {val_df['file_name'].nunique()}")
    print(f"Train 객체 수: {len(train_df)}")
    print(f"Val 객체 수: {len(val_df)}")
    print("Train 클래스 수:", train_df["class_id"].nunique())
    print("Val 클래스 수:", val_df["class_id"].nunique())

    # 겹치는 이미지가 없는지 확인
    overlap = set(train_imgs) & set(val_imgs)
    print(f"Train/Val 겹치는 이미지 수: {len(overlap)}")

    save_dir.mkdir(parents=True, exist_ok=True)

    train_ann_path = save_dir / "train_annotations.csv"
    val_ann_path = save_dir / "val_annotations.csv"
    train_img_path = save_dir / "train_images.csv"
    val_img_path = save_dir / "val_images.csv"

    train_df.to_csv(train_ann_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_ann_path, index=False, encoding="utf-8-sig")
    train_img_df.to_csv(train_img_path, index=False, encoding="utf-8-sig")
    val_img_df.to_csv(val_img_path, index=False, encoding="utf-8-sig")

    print("\ntrain/val split 저장 완료")
    print(train_ann_path)
    print(val_ann_path)
    print(train_img_path)
    print(val_img_path)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "train_img_df": train_img_df,
        "val_img_df": val_img_df,
        "train_ann_path": train_ann_path,
        "val_ann_path": val_ann_path,
        "train_img_path": train_img_path,
        "val_img_path": val_img_path,
    }


def main():
    master_csv = Path("data/processed/v1/master_annotations.csv")
    save_dir = Path("data/processed/v1")
    val_size = 0.2
    random_state = 42

    make_split(
        master_csv=master_csv,
        save_dir=save_dir,
        val_size=val_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    main()