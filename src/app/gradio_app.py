import gradio as gr
from app.model import AstronomicalClassifier
from models.models import Net


model = Net()
ckpt_path = "src/checkpoints/model_w_61.pth"
classifier = AstronomicalClassifier(model, ckpt_path)

def output_fn(input_img):
    return classifier.serve(input_img)


def run_app():
    demo = gr.Interface(
        fn=output_fn,
        inputs=gr.Image(type="numpy"),
        outputs=gr.components.Label(num_top_classes=3)
    )

    demo.launch()

