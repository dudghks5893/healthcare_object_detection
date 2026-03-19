from pathlib import Path
import pandas as pd

"""
    실행 순서: 6 (val 분리 없이 전부 학습 데이터로 사용 하고 싶을 시 사용)

    직접 실행: python -m src.preprocessing.make_stage2_fulltrain_csv

    [역할]
    Stage2 classifier 학습을 위해 train + val 데이터를 하나로 합치는 단계 (full-train)
    임시 파일. 차 후 train/val 나눠서 성능 개선 들어 갈 시 필요 없어짐.

    [하는 일]
    - train / val crop metadata CSV 병합
    - full-train 학습용 CSV 생성

    [결과]
    full_train_crop_labels.csv
"""

TRAIN_CSV = Path("data/processed/v1/stage2_classifier_crop_dataset/metadata/train_crop_labels.csv")
VAL_CSV = Path("data/processed/v1/stage2_classifier_crop_dataset/metadata/val_crop_labels.csv")
SAVE_CSV = Path("data/processed/v1/stage2_classifier_crop_dataset/metadata/full_train_crop_labels.csv")

def make_stage2_fulltrain_csv(
    train_csv: Path,
    val_csv: Path,
    save_csv: Path,
):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if train_df.empty or val_df.empty:
        print("\n[경고] train 또는 val CSV가 비어 있습니다.")
        print(f"- train_csv: {train_csv}")
        print(f"- val_csv: {val_csv}")
        return None

    full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    save_csv.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(save_csv, index=False, encoding="utf-8-sig")

    print("\n=== Stage2 Full-Train CSV 생성 완료 ===")
    print(f"저장 위치: {save_csv}")
    print(f"총 crop 수: {len(full_df)}")
    print(f"총 클래스 수: {full_df['class_id'].nunique()}")

    return {
        "full_df": full_df,
        "save_path": save_csv,
        "num_samples": len(full_df),
        "num_classes": full_df["class_id"].nunique(),
    }


def main():
    make_stage2_fulltrain_csv(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        save_csv=SAVE_CSV,
    )

if __name__ == "__main__":
    main()