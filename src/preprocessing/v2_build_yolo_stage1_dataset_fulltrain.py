import shutil
from pathlib import Path

import pandas as pd
import yaml


"""
실행 예시:
    python -m src.preprocessing.v2_build_yolo_stage1_dataset_fulltrain

[역할]
- v2 master_annotations.csv를 기반으로
  Stage 1 YOLO detector 학습용 full-train dataset을 생성하는 단계

[상황]
- 이번 v2 데이터는 train / val split 없이
  전체 데이터를 전부 train으로만 사용하여 학습할 예정
- 따라서 images/train, labels/train 만 생성
- validation metric은 학습 시 val=False로 끄는 것을 권장

[입력]
- data/processed/v2/master_annotations.csv
- data/raw/v2/train_images/

[출력]
- data/processed/v2/yolo_stage1_dataset_fulltrain/
    ├── images/train/
    ├── labels/train/
    └── data.yaml

[중요]
- Stage 1 detector의 목적은 "알약 위치 탐지"이므로
  모든 bbox label class는 단일 클래스 0 ("pill") 로 저장
"""


def make_yolo_label_line(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: float,
    img_h: float,
) -> str:
    """
    COCO bbox(x_min, y_min, width, height)를
    YOLO 형식(class_id, x_center, y_center, width, height)으로 변환한다.

    YOLO 포맷:
        class_id x_center y_center width height
    모든 좌표는 0~1 사이로 정규화된 값이어야 한다.

    Parameters
    ----------
    x, y, w, h : float
        COCO bbox 값
    img_w, img_h : float
        원본 이미지 너비 / 높이

    Returns
    -------
    str
        YOLO 형식 한 줄 문자열
    """
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Stage 1에서는 모든 알약을 하나의 클래스("pill")로 본다.
    class_idx = 0

    return f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def save_yolo_train_dataset(
    df: pd.DataFrame,
    raw_img_dir: Path,
    save_root: Path,
) -> None:
    """
    master dataframe을 기반으로 full-train YOLO dataset을 저장한다.

    생성되는 구조:
        save_root/
            ├── images/train/
            └── labels/train/

    같은 이미지(file_name)에 여러 bbox가 있을 수 있으므로
    file_name 기준으로 groupby 후,
    - 이미지는 1번 복사
    - label txt에는 해당 이미지의 bbox 여러 줄 저장
    """
    image_save_dir = save_root / "images" / "train"
    label_save_dir = save_root / "labels" / "train"

    image_save_dir.mkdir(parents=True, exist_ok=True)
    label_save_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby("file_name")

    missing_images = []
    invalid_bbox_count = 0
    total_images = 0
    total_labels = 0

    for file_name, group in grouped:
        src_img_path = raw_img_dir / file_name

        # 원본 이미지가 없으면 해당 샘플은 건너뛴다.
        if not src_img_path.exists():
            missing_images.append(str(src_img_path))
            continue

        # 이미지 복사
        dst_img_path = image_save_dir / file_name
        shutil.copy2(src_img_path, dst_img_path)

        # YOLO 라벨 txt 저장 경로
        label_path = label_save_dir / f"{Path(file_name).stem}.txt"
        lines = []

        for _, row in group.iterrows():
            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            w = float(row["bbox_w"])
            h = float(row["bbox_h"])
            img_w = float(row["width"])
            img_h = float(row["height"])

            # 비정상 bbox는 제외
            if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
                invalid_bbox_count += 1
                continue

            line = make_yolo_label_line(
                x=x,
                y=y,
                w=w,
                h=h,
                img_w=img_w,
                img_h=img_h,
            )
            lines.append(line)

        # bbox가 하나도 유효하지 않은 경우에도
        # 빈 txt 파일을 만들어 YOLO가 background 이미지로 인식하게 할 수 있다.
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        total_images += 1
        total_labels += len(lines)

    print("\n==== train 저장 결과 ====")
    print(f"이미지 수: {total_images}")
    print(f"라벨 수(bbox 수): {total_labels}")
    print(f"누락 이미지 수: {len(missing_images)}")
    print(f"제외된 비정상 bbox 수: {invalid_bbox_count}")

    if missing_images:
        print("\n[경고] 일부 이미지가 없어 복사되지 않았습니다.")
        for p in missing_images[:10]:
            print(p)
        if len(missing_images) > 10:
            print(f"... 외 {len(missing_images) - 10}개")


def build_v2_yolo_stage1_dataset_fulltrain(
    master_csv: Path,
    raw_img_dir: Path,
    save_root: Path,
) -> None:
    """
    v2 master_annotations.csv를 기반으로
    YOLO Stage 1 full-train dataset을 생성한다.

    Parameters
    ----------
    master_csv : Path
        v2 master_annotations.csv 경로
    raw_img_dir : Path
        원본 이미지 폴더 경로
    save_root : Path
        YOLO dataset 저장 루트 경로
    """
    if not master_csv.exists():
        raise FileNotFoundError(f"master csv가 없습니다: {master_csv}")

    if not raw_img_dir.exists():
        raise FileNotFoundError(f"raw image 폴더가 없습니다: {raw_img_dir}")

    df = pd.read_csv(master_csv, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("master csv가 비어 있습니다.")

    required_cols = [
        "file_name",
        "width",
        "height",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"master csv에 필수 컬럼이 없습니다: {missing_cols}")

    # 숫자 컬럼 강제 변환
    numeric_cols = ["width", "height", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 필수 숫자 컬럼 NaN 제거
    before_len = len(df)
    df = df.dropna(subset=numeric_cols).copy()
    dropped_len = before_len - len(df)

    print("\n==== v2 full-train Stage1 YOLO dataset 생성 ====")
    print("모든 bbox label class는 0 (pill)로 저장됩니다.")

    print("\n==== 입력 데이터 정보 ====")
    print(f"전체 행 수(annotation 수): {before_len}")
    print(f"NaN 제거 후 annotation 수: {len(df)}")
    print(f"제거된 annotation 수: {dropped_len}")
    print(f"전체 이미지 수: {df['file_name'].nunique()}")

    save_yolo_train_dataset(
        df=df,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
    )

    # full-train 전용 data.yaml 생성
    # val은 형식상 train과 동일하게 넣어두었지만,
    # 실제 학습에서는 val=False로 validation을 끄는 것을 권장한다.
    data_yaml = {
        "path": str(save_root.resolve()),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": ["pill"],
    }

    data_yaml_path = save_root / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data_yaml,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print("\n==== 저장 완료 ====")
    print(f"YOLO dataset root: {save_root}")
    print(f"data.yaml: {data_yaml_path}")

    train_img_count = len(list((save_root / "images" / "train").glob("*")))
    train_label_count = len(list((save_root / "labels" / "train").glob("*.txt")))

    print("\n==== 최종 검증 ====")
    print(f"train images: {train_img_count}")
    print(f"train labels: {train_label_count}")

    sample_train_labels = list((save_root / "labels" / "train").glob("*.txt"))
    if sample_train_labels:
        sample_label_path = sample_train_labels[0]
        print("\n==== 샘플 라벨 파일 확인 ====")
        print(f"sample label path: {sample_label_path}")
        with open(sample_label_path, "r", encoding="utf-8") as f:
            sample_lines = f.read().splitlines()[:5]

        if sample_lines:
            for line in sample_lines:
                print(line)
        else:
            print("(빈 라벨 파일 - background 이미지일 수 있음)")


def main() -> None:
    """
    스크립트 단독 실행용 엔트리포인트
    """
    master_csv = Path("data/processed/v2/master_annotations.csv")
    raw_img_dir = Path("data/raw/v2/train_images")
    save_root = Path("data/processed/v2/yolo_stage1_dataset_fulltrain")

    build_v2_yolo_stage1_dataset_fulltrain(
        master_csv=master_csv,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
    )


if __name__ == "__main__":
    main()