from pathlib import Path
import pandas as pd

"""
[역할]
master_annotations.csv에서
class_id / class_name / image_id / file_name 컬럼을 확인하는 테스트용 스크립트

[실행]
python -m tests.check_master_ann
"""


def check_master_ann(
    csv_path: Path = Path("data/processed/v1/master_annotations.csv"),
):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    target_cols = ["class_id", "class_name", "image_id", "file_name"]

    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV에 없는 컬럼이 있습니다: {missing_cols}")

    result_df = df[target_cols].copy()

    # print("\n==== 고유 조합 개수 ====")
    # print(result_df.drop_duplicates().shape[0])

    

    # print("\n==== class_id / class_name 매핑 ====")
    # class_map_df = (
    #     result_df[["class_id", "class_name"]]
    #     .drop_duplicates()
    #     .sort_values(["class_id", "class_name"])
    #     .reset_index(drop=True)
    # )
    # print(class_map_df)

    # print("\n==== image_id / file_name / class 정보 일부 ====")
    # preview_df = (
    #     result_df
    #     .drop_duplicates()
    #     .sort_values(["file_name", "class_id"])
    #     .reset_index(drop=True)
    # )
    # print(preview_df.head(50))
    # print("\n==== 클레스 종류 ====")
    # print(df["class_id"].drop_duplicates())

    # print("\n==== 클레스 종류222 ====")
    # print(df["class_id"].drop_duplicates().count())

    print("\n==== 클레스 종류333 ====")
    # print(df[["class_name", "class_id"]].drop_duplicates())
    print(df[["class_name", "class_id"]].drop_duplicates().count())

    return result_df


def main():
    check_master_ann()


if __name__ == "__main__":
    main()