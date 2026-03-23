import random
from pathlib import Path

import pandas as pd
from PIL import Image

from torchvision import transforms
from torchvision.utils import save_image


"""
    [역할]
    Stage2 augmentation 결과를 local에 저장해서 눈으로 확인

    [실행]
    python -m tests.preview_stage2_augmentation
"""


# =========================
# 설정
# =========================

CSV_PATH = Path(
    "data/processed/v1/stage2_classifier_crop_dataset/metadata/full_train_crop_labels.csv"
)

SAVE_DIR = Path(
    "data/processed/v1/stage2_classifier_crop_dataset/stage2_aug_preview"
)

IMG_SIZE = 224

NUM_IMAGES = 5
NUM_AUG_PER_IMAGE = 3

SEED = 42


# =========================
# Preview용 transform
# =========================

def build_preview_transform(img_size: int = 224):
    """
        미리보기 저장용:
        Normalize 없이, 눈으로 확인 가능한 PIL/Tensor 상태 유지
    """
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),

        # 1) 색/조명 변화 대응
        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.03,
            hue=0.0,
        ),

        # 2) 글자 선명도 변화 대응
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
            p=0.2,
        ),

        # 3) 위치/크기 변화만 아주 약하게
        transforms.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
        ),

        transforms.ToTensor(),
    ])
    return tf


# =========================
# 이미지 경로 컬럼 자동 탐색
# =========================

def find_image_path_column(df):

    candidates = [
        "image_path",
        "crop_path",
        "file_path",
        "img_path",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        "이미지 경로 컬럼을 찾지 못했습니다.\n"
        "가능 후보: image_path / crop_path / file_path / img_path"
    )


# =========================
# Preview 저장
# =========================

def save_preview():

    print("\n[Stage2 Augmentation Preview 시작]")

    random.seed(SEED)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    if len(df) == 0:
        raise ValueError("CSV가 비어 있습니다.")

    path_col = find_image_path_column(df)

    print("사용 컬럼:", path_col)

    preview_tf = build_preview_transform(IMG_SIZE)

    indices = random.sample(
        range(len(df)),
        k=min(NUM_IMAGES, len(df)),
    )

    for i, idx in enumerate(indices, start=1):

        row = df.iloc[idx]

        img_path = Path(row[path_col])

        if not img_path.exists():
            print("이미지 없음:", img_path)
            continue

        img = Image.open(img_path).convert("RGB")

        # ---------------------
        # 원본 저장
        # ---------------------

        orig = transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        )(img)

        orig_tensor = transforms.ToTensor()(orig)

        save_image(
            orig_tensor,
            SAVE_DIR / f"{i:02d}_orig.png",
        )

        # ---------------------
        # 증강 저장
        # ---------------------

        for j in range(1, NUM_AUG_PER_IMAGE + 1):

            aug = preview_tf(img)

            save_image(
                aug,
                SAVE_DIR / f"{i:02d}_aug{j}.png",
            )

    print("\n저장 완료:")
    print(SAVE_DIR)


# =========================
# main
# =========================

def main():

    save_preview()


if __name__ == "__main__":
    main()