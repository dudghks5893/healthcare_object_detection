from torchvision import transforms
from PIL import ImageFilter
import random


# 샤프닝
class RandomSharpen:
    """
        약하게 선명도를 올려 글자/각인 디테일 학습에 도움을 주는 transform
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.SHARPEN)
        return img


# 더 강한 샤프닝
class StrongSharpen:

    def __call__(self, img):

        return img.filter(
            ImageFilter.UnsharpMask(
                radius=2,
                percent=150,
                threshold=3
            )
        )


class RandomRotate180:
    """
        50% 확률로 180도 회전
        degrees=180처럼 모든 각도로 랜덤 회전하지 않고
        0도 / 180도만 적용
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.rotate(img, 180)
        return img