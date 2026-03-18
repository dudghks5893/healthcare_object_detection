from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

"""
    실행 순서: 2
    
    [역할]
    master CSV를 기반으로 train / validation 데이터를 분리하는 단계

    [하는 일]
    - 이미지 단위로 데이터 분할 (데이터 누수 방지)
    - train_annotations.csv / val_annotations.csv 생성

    [결과]
    data/processed/.../train_annotations.csv
    data/processed/.../val_annotations.csv
"""

MASTER_CSV = Path("data/processed/master_annotations.csv")
SAVE_DIR = Path("data/processed")
RANDOM_STATE = 42
VAL_SIZE = 0.2

df = pd.read_csv(MASTER_CSV)

print("==== master 정보 ====")
print(f"총 객체 수: {len(df)}")
print(f"총 이미지 수: {df['file_name'].nunique()}")
print(f"총 클래스 수: {df['class_id'].nunique()}")

# 이미지 단위 split
unique_images = df["file_name"].drop_duplicates().values

train_imgs, val_imgs = train_test_split(
    unique_images,
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
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

# 저장
SAVE_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(SAVE_DIR / "train_annotations.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(SAVE_DIR / "val_annotations.csv", index=False, encoding="utf-8-sig")
train_img_df.to_csv(SAVE_DIR / "train_images.csv", index=False, encoding="utf-8-sig")
val_img_df.to_csv(SAVE_DIR / "val_images.csv", index=False, encoding="utf-8-sig")

print("\n저장 완료")
print(SAVE_DIR / "train_annotations.csv")
print(SAVE_DIR / "val_annotations.csv")
print(SAVE_DIR / "train_images.csv")
print(SAVE_DIR / "val_images.csv")