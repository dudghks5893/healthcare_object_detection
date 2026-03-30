from torchvision import transforms
# from src.utils import RandomRotate180
# from src.utils import RandomSharpen

"""
    [역할]
    Stage2 classifier용 transform 생성 함수 모음

    [포함]
    - 학습/검증용 transform
    - augmentation preview용 transform
"""


def build_stage2_transforms(
    img_size: int = 224,
    train: bool = True,
    augmentation: dict | None = None,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if (not train) or (not augmentation) or (not augmentation.get("use_train_aug", False)):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

    tf_list = [transforms.Resize((img_size, img_size))]

    color_jitter_cfg = augmentation.get("color_jitter")
    if color_jitter_cfg:
        tf_list.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=color_jitter_cfg.get("brightness", 0.0),
                        contrast=color_jitter_cfg.get("contrast", 0.0),
                        saturation=color_jitter_cfg.get("saturation", 0.0),
                        hue=color_jitter_cfg.get("hue", 0.0),
                    )
                ],
                p=color_jitter_cfg.get("p", 1.0),
            )
        )

    horizontal_flip_cfg = augmentation.get("horizontal_flip")
    if horizontal_flip_cfg:
        tf_list.append(
            transforms.RandomHorizontalFlip(
                p=horizontal_flip_cfg.get("p", 0.0)
            )
        )

    vertical_flip_cfg = augmentation.get("vertical_flip")
    if vertical_flip_cfg:
        tf_list.append(
            transforms.RandomVerticalFlip(
                p=vertical_flip_cfg.get("p", 0.0)
            )
        )

    gaussian_blur_cfg = augmentation.get("gaussian_blur")
    if gaussian_blur_cfg:
        tf_list.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=gaussian_blur_cfg.get("kernel_size", 3),
                        sigma=tuple(gaussian_blur_cfg.get("sigma", [0.1, 0.5])),
                    )
                ],
                p=gaussian_blur_cfg.get("p", 0.0),
            )
        )

    random_affine_cfg = augmentation.get("random_affine")
    if random_affine_cfg:
        tf_list.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=random_affine_cfg.get("degrees", 0),
                        translate=tuple(random_affine_cfg.get("translate", [0.0, 0.0])),
                        scale=tuple(random_affine_cfg.get("scale", [1.0, 1.0])),
                    )
                ],
                p=random_affine_cfg.get("p", 1.0),
            )
        )
    
    # # 테스트 증강 강제 적용 적용 (글자 선명도)
    # tf_list.append(
    #         RandomSharpen(p=0.5),
    #     )
    # # 테스트 증강 강제 적용 적용 (0or180 도로 기울기)
    # tf_list.append(
    #         RandomRotate180(p=0.5),
    #     )
    
    tf_list.extend([
        transforms.ToTensor(),
        normalize,
    ])

    return transforms.Compose(tf_list)


def build_stage2_preview_transform(
    img_size: int = 224,
    augmentation: dict | None = None,
):
    if not augmentation or not augmentation.get("use_train_aug", False):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    tf_list = [transforms.Resize((img_size, img_size))]

    color_jitter_cfg = augmentation.get("color_jitter")
    if color_jitter_cfg:
        tf_list.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=color_jitter_cfg.get("brightness", 0.0),
                        contrast=color_jitter_cfg.get("contrast", 0.0),
                        saturation=color_jitter_cfg.get("saturation", 0.0),
                        hue=color_jitter_cfg.get("hue", 0.0),
                    )
                ],
                p=color_jitter_cfg.get("p", 1.0),
            )
        )

    horizontal_flip_cfg = augmentation.get("horizontal_flip")
    if horizontal_flip_cfg:
        tf_list.append(
            transforms.RandomHorizontalFlip(
                p=horizontal_flip_cfg.get("p", 0.0)
            )
        )

    vertical_flip_cfg = augmentation.get("vertical_flip")
    if vertical_flip_cfg:
        tf_list.append(
            transforms.RandomVerticalFlip(
                p=vertical_flip_cfg.get("p", 0.0)
            )
        )

    gaussian_blur_cfg = augmentation.get("gaussian_blur")
    if gaussian_blur_cfg:
        tf_list.append(
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=gaussian_blur_cfg.get("kernel_size", 3),
                        sigma=tuple(gaussian_blur_cfg.get("sigma", [0.1, 0.5])),
                    )
                ],
                p=gaussian_blur_cfg.get("p", 0.0),
            )
        )

    random_affine_cfg = augmentation.get("random_affine")
    if random_affine_cfg:
        tf_list.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=random_affine_cfg.get("degrees", 0),
                        translate=tuple(random_affine_cfg.get("translate", [0.0, 0.0])),
                        scale=tuple(random_affine_cfg.get("scale", [1.0, 1.0])),
                    )
                ],
                p=random_affine_cfg.get("p", 1.0),
            )
        )

    tf_list.append(transforms.ToTensor())
    return transforms.Compose(tf_list)