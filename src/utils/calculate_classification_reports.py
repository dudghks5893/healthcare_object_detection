from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 총 평가 지표
def get_calculate_classification_reports(y_true, y_pred):

    print("=" * 60)
    print("STAGE2 CLASSIFICATION EVALUATION")
    print("=" * 60)

    acc = accuracy_score(y_true, y_pred)

    recall = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report")
    print(classification_report(y_true, y_pred))

# 간단 성능 지표 (매 에폭마다)
def get_calculate_metrics(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)

    recall = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return acc, recall, f1