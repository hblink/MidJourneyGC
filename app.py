import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
import numpy as np

model_id = "Joeythemonster/anything-midjourney-v-4-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")


def generate_image(prompt):
    image = pipe(prompt).images[0]
    image = np.asarray(image)
    return image

iface = gr.Interface(generate_image, gr.inputs.Textbox(label="Enter a prompt"), gr.outputs.Image(type='numpy'), batch_size=1)

iface.launch()
