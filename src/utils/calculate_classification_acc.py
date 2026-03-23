import torch
from torchvision.ops import box_iou

def calculate_classification_acc(predictions, targets, iou_threshold=0.5, score_threshold=0.3):
    total = 0
    correct = 0

    for pred, tgt in zip(predictions, targets):
        # ---------------------------------
        # 1. score threshold filtering
        # ---------------------------------
        keep = pred["scores"] > score_threshold
        pred_boxes = pred["boxes"][keep]
        pred_labels = pred["labels"][keep]

        tgt_boxes = tgt["boxes"]
        tgt_labels = tgt["labels"]

        if len(pred_boxes) == 0 or len(tgt_boxes) == 0:
            continue

        # ---------------------------------
        # 2. IoU 계산
        # ---------------------------------
        ious = box_iou(tgt_boxes, pred_boxes)  # (num_gt, num_pred)

        # ---------------------------------
        # 3. GT 기준으로 매칭
        # ---------------------------------
        for i in range(len(tgt_boxes)):
            iou_row = ious[i]

            max_iou, max_idx = torch.max(iou_row, dim=0)

            if max_iou >= iou_threshold:
                total += 1

                if pred_labels[max_idx] == tgt_labels[i]:
                    correct += 1

    # ---------------------------------
    # 4. accuracy 계산
    # ---------------------------------
    if total == 0:
        return 0.0

    return correct / total