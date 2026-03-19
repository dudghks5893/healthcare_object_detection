# ResNet 모델 Fine-Tuning 함수
def set_fine_tuning(model, mode="frozen"):
    """
        mode:
        - "frozen"
        - "partial"
        - "full"
    """

    # 기본 parameters 전부 freeze
    for param in model.parameters():
        param.requires_grad = False

    if mode == "frozen":
        # FC만 학습
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "partial":
        # layer4 + FC 학습
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "full":
        # 전부 학습
        for param in model.parameters():
            param.requires_grad = True

    else:
        raise ValueError("mode must be 'frozen', 'partial', or 'full'")