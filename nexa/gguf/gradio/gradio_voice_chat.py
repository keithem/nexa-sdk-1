import gradio as gr
from nexa.gguf.nexa_inference_voice import NexaVoiceInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model
import argparse

nexa_model = None

def load_model(model_path, is_local=False, hf=False, local_model_path=None):
    if is_local:
        local_path = local_model_path
    elif hf:
        local_path, _ = pull_model(model_path, hf=True)
    else:
        local_path, _ = pull_model(model_path)
    nexa_model = NexaVoiceInference(
        model_path=model_path,
        local_path=local_path
    )
    return nexa_model

def process_audio_fn(audio_file, beam_size, task, temperature):
    if not nexa_model:
        return "No model loaded. Please ensure the model is correctly initialized."
    if not audio_file:
        return "Please provide an audio input."

    try:
        with suppress_stdout_stderr():
            segments, _ = nexa_model.model.transcribe(
                audio_file,
                beam_size=beam_size,
                task=task,
                temperature=temperature,
                vad_filter=True
            )
        transcription = "".join(segment.text for segment in segments)
        return transcription
    except Exception as e:
        return f"Error during audio transcription: {e}"

with gr.Blocks() as demo:
    gr.HTML("...title and HTML same as before...")

    with gr.Row():
        beam_size_slider = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Beam Size")
        task_dropdown = gr.Dropdown(choices=["transcribe", "translate"], value="transcribe", label="Task")
        temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Temperature")

    gr.HTML("<h2>Option 1: Upload Audio</h2>")
    with gr.Row():
        uploaded_audio = gr.Audio(type="filepath", label="Upload an audio file (wav, mp3)")
        upload_response = gr.Textbox(label="Model Response", interactive=False)
    process_upload_button = gr.Button("Process Uploaded Audio")
    process_upload_button.click(process_audio_fn,
                                inputs=[uploaded_audio, beam_size_slider, task_dropdown, temperature_slider],
                                outputs=upload_response)

    gr.HTML("<h2>Option 2: Real-time Recording</h2>")
    recorded_audio = gr.Audio(source="microphone", type="filepath", label="Record Audio from Microphone")
    record_response = gr.Textbox(label="Model Response", interactive=False)
    process_record_button = gr.Button("Process Recorded Audio")
    process_record_button.click(process_audio_fn,
                                inputs=[recorded_audio, beam_size_slider, task_dropdown, temperature_slider],
                                outputs=record_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model name or path to load")
    parser.add_argument("--local_path", action="store_true", help="Use local path directly")
    parser.add_argument("--huggingface", action="store_true", help="Pull model from Hugging Face")
    parser.add_argument("--modelscope", action="store_true", help="Pull model from ModelScope (not shown in example)")
    parser.add_argument("--local_model_path", type=str, default=None, help="Local model folder or file path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Whether to share the Gradio app publicly")
    args = parser.parse_args()

    nexa_model = load_model(
        model_path=args.model_path,
        is_local=args.local_path,
        hf=args.huggingface,
        local_model_path=args.local_model_path
    )
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
