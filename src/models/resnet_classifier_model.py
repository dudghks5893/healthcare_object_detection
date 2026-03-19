import torch.nn as nn
from torchvision import models


class ResNetClassifierModel(nn.Module):
    """
        ResNet 기반 분류 모델

        지원 모델:
        - resnet18
        - resnet34
        - resnet50
        - resnet101
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "resnet18",
        pretrained: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone = self._build_backbone(model_name, pretrained)

        # 마지막 FC 레이어 교체
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _build_backbone(self, model_name: str, pretrained: bool):
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)

        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)

        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)

        elif model_name == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            model = models.resnet101(weights=weights)

        else:
            raise ValueError(f"지원하지 않는 model_name: {model_name}")

        return model

    def forward(self, x):
        return self.backbone(x)
