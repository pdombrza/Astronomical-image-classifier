import gradio as gr
from app.model import ModelStub


def output_fn(input_img):
    classifier = ModelStub()
    return classifier.serve(input_img)


def run_app():
    demo = gr.Interface(
        fn=output_fn,
        inputs=gr.Image(),  # Gradio expects an image as input
        outputs=gr.components.Label(num_top_classes=3)   # Outputs a string message
    )

    demo.launch()

