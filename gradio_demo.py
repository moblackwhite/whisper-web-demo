import numpy as np
import whisper
import gradio as gr
import ffmpeg


def extract_to_mp3(file_path):
    ffmpeg.input(file_path).output('audio.mp3').run()
    return "audio.mp3"


def extract_to_text(audio, model):
    model = whisper.load_model(model)
    result = model.transcribe(audio, word_timestamps=True)
    return result["text"]


with gr.Blocks() as demo:
    with gr.Tab("Extract mp3"):
        video_input = gr.Video()
        audio_output = gr.Audio()
        text_button = gr.Button("Transform")
    text_button.click(extract_to_mp3, inputs=video_input, outputs=audio_output)

    with gr.Tab("Extract mp3"):
        # extract_to_text_button.click(extract_to_text, inputs=(audio_input, model_choice), outputs=text_output)
        gr.Interface(
            extract_to_text,
            [
                gr.Audio(type='filepath'),
                gr.Dropdown(
                    ["tiny", "base", "small", "medium", "large"],
                    value=0,
                    label="Size"
                ),
            ],
            "text"
        )

demo.launch()
