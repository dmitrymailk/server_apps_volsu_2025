import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, Iterable
import fitz
import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

import shlex
import subprocess


MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Qwen3-VL-4B-Instruct
MODEL_ID_Q = "Qwen/Qwen3-VL-4B-Instruct"
processor_q = AutoProcessor.from_pretrained(MODEL_ID_Q, trust_remote_code=True)
model_q = (
    Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID_Q, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    .to(device)
    .eval()
)


def convert_pdf_to_images(file_path: str, dpi: int = 128):
    """
    Открывает все страницы PDF и преобразует их в картинки
    которые затем идут в LLM
    """
    if not file_path:
        return []
    images = []
    pdf_document = fitz.open(file_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        images.append(Image.open(BytesIO(img_data)))
    pdf_document.close()
    return images


def get_initial_pdf_state() -> Dict[str, Any]:
    return {
        "pages": [],
        "total_pages": 0,
        "current_page_index": 0,
    }


def load_and_preview_pdf(
    file_path: Optional[str],
) -> Tuple[Optional[Image.Image], Dict[str, Any], str]:
    state = get_initial_pdf_state()
    if not file_path:
        return None, state, '<div style="text-align:center;">No file loaded</div>'
    try:
        pages = convert_pdf_to_images(file_path)
        if not pages:
            return (
                None,
                state,
                '<div style="text-align:center;">Could not load file</div>',
            )
        state["pages"] = pages
        state["total_pages"] = len(pages)
        page_info_html = (
            f'<div style="text-align:center;">Page 1 / {state["total_pages"]}</div>'
        )
        return pages[0], state, page_info_html
    except Exception as e:
        return (
            None,
            state,
            f'<div style="text-align:center;">Failed to load preview: {e}</div>',
        )


def navigate_pdf_page(direction: str, state: Dict[str, Any]):
    """
    Слайдер для загруженных картинок из PDF
    """
    if not state or not state["pages"]:
        return None, state, '<div style="text-align:center;">No file loaded</div>'
    current_index = state["current_page_index"]
    total_pages = state["total_pages"]
    if direction == "prev":
        new_index = max(0, current_index - 1)
    elif direction == "next":
        new_index = min(total_pages - 1, current_index + 1)
    else:
        new_index = current_index
    state["current_page_index"] = new_index
    image_preview = state["pages"][new_index]
    page_info_html = (
        f'<div style="text-align:center;">Page {new_index + 1} / {total_pages}</div>'
    )
    return image_preview, state, page_info_html


def downsample_video(video_path, max_dim=720):
    """
    открывает видео и выбирает только некоторые кадры из видео
    и уменьшает его разрешение
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, min(total_frames, 10), dtype=int)

    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]
            scale = max_dim / max(h, w)
            if scale < 1:
                image = cv2.resize(
                    image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            pil_image = Image.fromarray(image)
            frames.append(pil_image)

    vidcap.release()
    return frames


def generate_image(
    model_name: str,
    text: str,
    image: Image.Image,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Generates responses using the selected model for image input.
    """
    if model_name == "Qwen3-VL-4B-Instruct":
        processor, model = processor_q, model_q
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return
    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return
    # создаем вид промпт для подачи в модель
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    # конвертируем его в текст
    prompt_full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # конвертируем его в токены и тензоры
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)
    # эта функция позволяет видеть генерация в реальном времени
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
    }
    # создаем неблокирующий тред, чтобы моментально после генерации токена
    # обновлять контент на странице
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer


def generate_video(
    model_name: str,
    text: str,
    video_path: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Generates responses using the selected model for video input.
    """
    if model_name == "Qwen3-VL-4B-Instruct":
        processor, model = processor_q, model_q
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return
    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return
    frames = downsample_video(video_path)
    if not frames:
        yield "Could not process video.", "Could not process video."
        return
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    images_for_processor = []
    for frame in frames:
        messages[0]["content"].append({"type": "image"})
        images_for_processor.append(frame)
    prompt_full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt_full],
        images=images_for_processor,
        return_tensors="pt",
        padding=True,
    ).to(device)
    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer


def generate_pdf(
    model_name: str,
    text: str,
    state: Dict[str, Any],
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):

    if model_name == "Qwen3-VL-4B-Instruct":
        processor, model = processor_q, model_q
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if not state or not state["pages"]:
        yield "Please upload a PDF file first.", "Please upload a PDF file first."
        return

    page_images = state["pages"]

    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    images_for_processor = []
    for frame in page_images:
        messages[0]["content"].append({"type": "image"})
        images_for_processor.append(frame)

    prompt_full = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt_full],
        images=images_for_processor,
        return_tensors="pt",
        padding=True,
    ).to(device)

    streamer = TextIteratorStreamer(
        processor, skip_prompt=True, skip_special_tokens=True
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")  # Thêm dòng này giống video
        yield buffer, buffer
        time.sleep(0.01)


image_examples = [
    ["Explain the content in detail.", "images/force.jpg"],
    ["Explain the content (ocr).", "images/ocr.jpg"],
    ["Extract the content in the json format", "images/bill.jpg"],
    ["Extract text from this image(ocr).", "images/handwritten.jpg"],
]
video_examples = [
    ["Identify the main actions in the video", "videos/2.mp4"],
]
pdf_examples = [
    ["Extract the content precisely.", "pdfs/doc1.pdf"],
]

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:

    pdf_state = gr.State(value=get_initial_pdf_state())

    gr.Markdown("# Распознавание картинок, PDF и видео.", elem_id="main-title")
    # создаем интерфейс на основе gradio
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(
                        label="Query Input", placeholder="Enter your query here..."
                    )
                    image_upload = gr.Image(
                        type="pil", label="Upload Image", height=290
                    )
                    image_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(
                        examples=image_examples, inputs=[image_query, image_upload]
                    )

                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(
                        label="Query Input", placeholder="Enter your query here..."
                    )
                    video_upload = gr.Video(label="Upload Video", height=290)
                    video_submit = gr.Button("Submit", variant="primary")
                    gr.Examples(
                        examples=video_examples, inputs=[video_query, video_upload]
                    )

                with gr.TabItem("PDF Inference"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            pdf_query = gr.Textbox(
                                label="Query Input",
                                placeholder="e.g., 'Summarize this document'",
                            )
                            pdf_upload = gr.File(
                                label="Upload PDF", file_types=[".pdf"]
                            )
                            pdf_submit = gr.Button("Submit", variant="primary")
                        with gr.Column(scale=1):
                            pdf_preview_img = gr.Image(label="PDF Preview", height=290)
                            with gr.Row():
                                prev_page_btn = gr.Button("◀ Previous")
                                page_info = gr.HTML(
                                    '<div style="text-align:center;">No file loaded</div>'
                                )
                                next_page_btn = gr.Button("Next ▶")
                    gr.Examples(examples=pdf_examples, inputs=[pdf_query, pdf_upload])

            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(
                    label="Max new tokens",
                    minimum=1,
                    maximum=MAX_MAX_NEW_TOKENS,
                    step=1,
                    value=DEFAULT_MAX_NEW_TOKENS,
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6
                )
                top_p = gr.Slider(
                    label="Top-p (nucleus sampling)",
                    minimum=0.05,
                    maximum=1.0,
                    step=0.05,
                    value=0.9,
                )
                top_k = gr.Slider(
                    label="Top-k", minimum=1, maximum=1000, step=1, value=50
                )
                repetition_penalty = gr.Slider(
                    label="Repetition penalty",
                    minimum=1.0,
                    maximum=2.0,
                    step=0.05,
                    value=1.2,
                )

        with gr.Column(scale=3):
            gr.Markdown("## Output", elem_id="output-title")
            output = gr.Textbox(
                label="Raw Output Stream",
                interactive=False,
                lines=14,
                show_copy_button=True,
            )
            with gr.Accordion("(Result.md)", open=False):
                markdown_output = gr.Markdown(
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ]
                )

            model_choice = gr.Radio(
                choices=[
                    "Qwen3-VL-4B-Instruct",
                ],
                label="Select Model",
                value="Qwen3-VL-4B-Instruct",
            )

    image_submit.click(
        fn=generate_image,
        inputs=[
            model_choice,
            image_query,
            image_upload,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        ],
        outputs=[output, markdown_output],
    )

    video_submit.click(
        fn=generate_video,
        inputs=[
            model_choice,
            video_query,
            video_upload,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        ],
        outputs=[output, markdown_output],
    )

    pdf_submit.click(
        fn=generate_pdf,
        inputs=[
            model_choice,
            pdf_query,
            pdf_state,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        ],
        outputs=[output, markdown_output],
    )

    pdf_upload.change(
        fn=load_and_preview_pdf,
        inputs=[pdf_upload],
        outputs=[pdf_preview_img, pdf_state, page_info],
    )

    prev_page_btn.click(
        fn=lambda s: navigate_pdf_page("prev", s),
        inputs=[pdf_state],
        outputs=[pdf_preview_img, pdf_state, page_info],
    )

    next_page_btn.click(
        fn=lambda s: navigate_pdf_page("next", s),
        inputs=[pdf_state],
        outputs=[pdf_preview_img, pdf_state, page_info],
    )


if __name__ == "__main__":
    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True)
