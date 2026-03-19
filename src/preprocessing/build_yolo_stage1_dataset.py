from pathlib import Path
import pandas as pd
import shutil

"""
    실행 순서: 3
    직접 실행: python -m src.preprocessing.build_yolo_stage1_dataset

    [역할]
    Stage1 (객체 탐지) 학습을 위한 YOLO 형식 데이터셋을 생성하는 단계

    [하는 일]
    1.train_annotations.csv, val_annotations.csv 읽음
    2.이미지별 bbox 묶음
    3.bbox를 YOLO 포맷 (x_center, y_center, w, h)으로 변환
    4.이미지와 label(.txt) 파일 생성(YOLO 형식 txt)
    5.원본 이미지 복사
    6.data.yaml 생성
    [결과]
    data/processed/.../yolo_stage1/
"""

def convert_to_yolo(df, image_out_dir: Path, label_out_dir: Path, raw_img_dir: Path):
    grouped = df.groupby("file_name")
    saved_image_count = 0
    saved_label_count = 0

    for file_name, group in grouped:
        img_path = raw_img_dir / file_name
        if not img_path.exists():
            print(f"[경고] 이미지 없음: {img_path}")
            continue

        # 이미지 복사
        shutil.copy2(img_path, image_out_dir / file_name)
        saved_image_count += 1

        # 같은 이미지의 bbox들을 하나의 txt로 저장
        label_path = label_out_dir / f"{Path(file_name).stem}.txt"

        lines = []
        for _, row in group.iterrows():
            img_w = float(row["width"])
            img_h = float(row["height"])

            x = float(row["bbox_x"])
            y = float(row["bbox_y"])
            w = float(row["bbox_w"])
            h = float(row["bbox_h"])

            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Stage 1은 알약 1클래스
            class_id = 0

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            lines.append(line)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        saved_label_count += 1

    return saved_image_count, saved_label_count


def build_yolo_stage1_dataset(
    train_csv: Path,
    val_csv: Path,
    raw_img_dir: Path,
    save_root: Path,
):
    train_img_out = save_root / "images" / "train"
    val_img_out = save_root / "images" / "val"
    train_label_out = save_root / "labels" / "train"
    val_label_out = save_root / "labels" / "val"

    for p in [train_img_out, val_img_out, train_label_out, val_label_out]:
        p.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if train_df.empty or val_df.empty:
        print("\n[경고] train 또는 val CSV가 비어 있습니다.")
        print(f"- train_csv: {train_csv}")
        print(f"- val_csv: {val_csv}")
        return None

    train_img_count, train_label_count = convert_to_yolo(
        train_df, train_img_out, train_label_out, raw_img_dir
    )
    val_img_count, val_label_count = convert_to_yolo(
        val_df, val_img_out, val_label_out, raw_img_dir
    )

    # data.yaml 생성
    yaml_text = f"""path: {save_root.as_posix()}
train: images/train
val: images/val

names:
  0: pill
"""

    yaml_path = save_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

    print("\nYOLO Stage1 데이터셋 생성 완료")
    print(save_root)

    print("\n==== 생성 요약 ====")
    print(f"Train 이미지 수: {train_img_count}")
    print(f"Train 라벨 수: {train_label_count}")
    print(f"Val 이미지 수: {val_img_count}")
    print(f"Val 라벨 수: {val_label_count}")
    print(f"data.yaml: {yaml_path}")

    # 샘플 라벨 출력
    label_files = list((save_root / "labels" / "train").glob("*.txt"))

    print("\n==== 샘플 YOLO 라벨 확인 ====")
    for lf in label_files[:3]:
        print(f"\n[{lf.name}]")
        with open(lf, "r", encoding="utf-8") as f:
            print(f.read())

    return {
        "train_img_out": train_img_out,
        "val_img_out": val_img_out,
        "train_label_out": train_label_out,
        "val_label_out": val_label_out,
        "yaml_path": yaml_path,
    }


def main():
    train_csv = Path("data/processed/v1/train_annotations.csv")
    val_csv = Path("data/processed/v1/val_annotations.csv")
    raw_img_dir = Path("data/raw/v1/train_images")
    save_root = Path("data/processed/v1/yolo_stage1_dataset")

    build_yolo_stage1_dataset(
        train_csv=train_csv,
        val_csv=val_csv,
        raw_img_dir=raw_img_dir,
        save_root=save_root,
    )


if __name__ == "__main__":
    main()