import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class PillDetectionDataset(Dataset):
    def __init__(
        self,
        file_list,
        image_dir,
        annotation_map=None,
        class_to_idx=None,
        transform=None,
        model_type="ssd",
    ):
        self.file_list = file_list
        self.image_dir = image_dir
        self.annotation_map = annotation_map or {}
        self.class_to_idx = class_to_idx or {}
        self.transform = transform
        self.model_type = model_type.lower()

        supported_types = ["ssd", "fasterrcnn", "retinanet", "yolo", "classification"]
        if self.model_type not in supported_types:
            raise ValueError(f"지원하지 않는 model_type: {self.model_type}")

    def __len__(self):
        return len(self.file_list)

    def _load_image(self, file_name):
        possible_paths = [
            os.path.join(self.image_dir, file_name + ".png"),
            os.path.join(self.image_dir, file_name + ".jpg"),
            os.path.join(self.image_dir, file_name + ".jpeg"),
        ]

        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break

        if img_path is None:
            raise FileNotFoundError(f"이미지 없음: {possible_paths}")

        image = Image.open(img_path).convert("RGB")
        return image

    def _parse_json(self, file_name):
        """
        현재 데이터셋은 COCO 스타일 JSON 구조를 사용한다고 가정

        예시:
        {
            "annotations": [
                {
                    "bbox": [x, y, width, height],
                    "category_id": 3
                }
            ],
            "categories": [
                {
                    "id": 3,
                    "name": "pill_name"
                }
            ]
        }
        """
        if file_name not in self.annotation_map:
            raise FileNotFoundError(f"annotation_map에 없는 파일입니다: {file_name}")

        json_path = self.annotation_map[file_name]

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        boxes = []
        labels = []

        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        # category_id -> category_name 매핑
        cat_id_to_name = {}
        for cat in categories:
            cat_id = cat.get("id")
            cat_name = cat.get("name")
            if cat_id is not None and cat_name is not None:
                cat_id_to_name[cat_id] = cat_name

        for ann in annotations:
            bbox = ann.get("bbox")          # [x, y, w, h]
            category_id = ann.get("category_id")

            if bbox is None or category_id is None:
                continue

            if category_id not in cat_id_to_name:
                continue

            cls_name = cat_id_to_name[category_id]

            if cls_name not in self.class_to_idx:
                continue

            x, y, w, h = map(float, bbox)

            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[cls_name])

        return boxes, labels

    def _build_torchvision_target(self, boxes, labels, idx):
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return target

    def _convert_xyxy_to_yolo(self, boxes, labels, image_width, image_height):
        yolo_targets = []

        for box, cls_id in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box

            x_center = ((xmin + xmax) / 2.0) / image_width
            y_center = ((ymin + ymax) / 2.0) / image_height
            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height

            yolo_targets.append([cls_id, x_center, y_center, width, height])

        if len(yolo_targets) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(yolo_targets, dtype=torch.float32)

    def _build_classification_target(self, labels):
        if len(labels) == 0:
            raise ValueError("classification 모드인데 유효한 label이 없습니다.")
        return torch.tensor(labels[0], dtype=torch.long)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]

        image = self._load_image(file_name)
        image_width, image_height = image.size

        boxes, labels = self._parse_json(file_name)

        if self.model_type != "classification" and len(boxes) == 0:
            raise ValueError(f"유효한 bbox가 없는 샘플입니다: {file_name}")

        if self.model_type in ["ssd", "fasterrcnn", "retinanet"]:
            target = self._build_torchvision_target(boxes, labels, idx)

            if self.transform is not None:
                image, target = self.transform(image, target)

            return image, target

        elif self.model_type == "yolo":
            if self.transform is not None:
                image = self.transform(image)

            if isinstance(image, torch.Tensor):
                _, h, w = image.shape
                image_width, image_height = w, h
            else:
                image_width, image_height = image.size

            yolo_target = self._convert_xyxy_to_yolo(
                boxes, labels, image_width, image_height
            )
            return image, yolo_target

        elif self.model_type == "classification":
            if self.transform is not None:
                image = self.transform(image)

            label = self._build_classification_target(labels)
            return image, label


def get_torchvision_train_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    ])


def get_torchvision_valid_transform():
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets) 