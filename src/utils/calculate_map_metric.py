from torchmetrics.detection.mean_ap import MeanAveragePrecision

# mAP 계산 함수
def calculate_map_metric(predictions, targets, score_threshold = 0.5):
        filtered_predictions = []

        for pred in predictions:
            # Confidence Score 임계값
            keep = pred["scores"] > score_threshold
            filtered_predictions.append({
                "boxes": pred["boxes"][keep],
                "scores": pred["scores"][keep],
                "labels": pred["labels"][keep],
            })

        metric = MeanAveragePrecision(iou_type="bbox")
        metric.update(filtered_predictions, targets)
        results = metric.compute()

        return results["map"].item()