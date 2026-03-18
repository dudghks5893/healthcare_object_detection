from pathlib import Path
import pandas as pd

"""
    실행 방법:
    터미널에 python -m src.preprocessing.make_stage2_fulltrain_csv 입력
"""

TRAIN_CSV = Path("data/processed/stage2_classifier/metadata/train_crop_labels.csv")
VAL_CSV = Path("data/processed/stage2_classifier/metadata/val_crop_labels.csv")
SAVE_CSV = Path("data/processed/stage2_classifier/metadata/full_train_crop_labels.csv")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

SAVE_CSV.parent.mkdir(parents=True, exist_ok=True)
full_df.to_csv(SAVE_CSV, index=False, encoding="utf-8-sig")

print(f"저장 완료: {SAVE_CSV}")
print(f"총 crop 수: {len(full_df)}")
print(f"총 클래스 수: {full_df['class_id'].nunique()}")