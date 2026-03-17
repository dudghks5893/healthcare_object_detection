'''
    테스트 방법:
    터미널에 python -m dataset_class 입력
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches # 도형 그리는 라이브러리
import pandas as pd
from PIL import Image

import numpy as np

# kaggle 데이터 받기
import kagglehub

# 데이터 복사
import shutil

# 운영체제(OS)와 파일 경로를 다루기 위한 라이브러리
import os

# dict 처리 간편화
from collections import defaultdict, Counter

# 프레임워크 PyTorch
import torch

# 시간
import time

# 해쉬 암호 알고리즘
import hashlib

# xml
import xml.etree.ElementTree as ET

# 데이터 셋 분할
from sklearn.model_selection import train_test_split

# Progress Bar
from tqdm import tqdm


# PyTorch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR # lr 스케쥴러
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.ops import box_iou # IoU 평가 지표

'''
# mAP 평가 지표
!pip install torchmetrics
'''
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class PillClassificationDataset(Dataset):
    def __init__(self, file_list, image_dir, label_dict, transform=None):
        self.file_list = file_list
        self.image_dir = image_dir
        self.label_dict = label_dict
        self.class_to_idx = {cls: i for i, cls in enumerate(set(label_dict.values()))}
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.image_dir, file_name + ".png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"이미지 없음: {img_path}")

        image = Image.open(img_path).convert("RGB")

        label_name = self.label_dict[file_name]
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)