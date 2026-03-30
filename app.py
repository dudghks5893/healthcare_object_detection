from pathlib import Path
import gradio as gr

from src.utils import load_config
from main import build_paths
from src.engine.predict_2stage import predict_one_image

'''
    실행 방법: python app.py
'''

CONFIG_PATH = "configs/yolo11s_resnet50_tr_ep_20_claasweights_deg_180_v4.yaml"

# ------------------------------------------
# 앱 시작 시 yaml 읽고 경로/설정 계산
# ------------------------------------------
cfg = load_config(CONFIG_PATH)
paths = build_paths(cfg)

detector_weight = paths["detector_weight"]
classifier_weight = paths["classifier_weight"]
classifier_dir = paths["stage2_checkpoint_dir"]

det_imgsz = cfg["stage1"]["imgsz"]
det_conf = cfg["stage1"]["conf"]
det_iou = cfg["stage1"]["iou"]
crop_margin_ratio = cfg["stage2"]["crop_margin_ratio"]


def run_prediction(image):
    if image is None:
        return None, "이미지를 업로드하세요."

    vis_img, predictions = predict_one_image(
        input_image=image,
        detector_weight=detector_weight,
        classifier_weight=classifier_weight,
        classifier_dir=classifier_dir,
        det_imgsz=det_imgsz,
        det_conf=det_conf,
        det_iou=det_iou,
        crop_margin_ratio=crop_margin_ratio,
    )

    if len(predictions) == 0:
        return vis_img, "검출된 알약이 없습니다."

    lines = []
    for i, pred in enumerate(predictions, start=1):
        lines.append(
            f"{i}. class={pred['category_id']} "
            f"score={pred['score']:.4f} "
            f"bbox={pred['bbox']}"
        )

    return vis_img, "\n".join(lines)


demo = gr.Interface(
    fn=run_prediction,
    inputs=gr.Image(type="pil", label="이미지 업로드"),
    outputs=[
        gr.Image(type="pil", label="예측 결과"),
        gr.Textbox(label="예측 상세"),
    ],
    title="Pill Detection Demo",
    description="이미지를 업로드하면 Stage1 + Stage2 모델로 예측합니다.",
)

if __name__ == "__main__":
    demo.launch()