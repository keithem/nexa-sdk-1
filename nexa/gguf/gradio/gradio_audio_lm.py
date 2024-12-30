import gradio as gr
from nexa.gguf.nexa_inference_audio_lm import NexaAudioLMInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model
import argparse
import sys

nexa_model = None

def load_model(model_path, is_local=False, hf=False, local_model_path=None, projector_local_path=None):
    """Load the AudioLM model from local path or HF or Nexa Hub."""
    if is_local:
        
        local_path = local_model_path
    elif hf:
        
        local_path, _ = pull_model(model_path, hf=True)
    else:
        
        local_path, _ = pull_model(model_path)
    
    nexa_model = NexaAudioLMInference(
        model_path=model_path,
        local_path=local_path,
        projector_local_path=projector_local_path
    )
    return nexa_model

def process_audio_fn(audio_file, prompt=""):
    if not nexa_model:
        return "No model loaded. Please load a model first."
    if not audio_file:
        return "Please provide an audio input."

    try:
        with suppress_stdout_stderr():
            response = nexa_model.inference(audio_file, prompt)
        return response
    except Exception as e:
        return f"Error during audio processing: {e}"

with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="display: flex; align-items: center; margin-bottom: 5px; padding-top: 10px;">
            <h1 style="font-family: Arial, sans-serif; font-size: 2.5em; font-weight: bold; margin: 0; padding-bottom: 5px;">
                Nexa AI AudioLM Generation
            </h1>
            <a href='https://github.com/NexaAI/nexa-sdk' style='text-decoration: none; margin-left: 10px;'>
                <img src='https://img.shields.io/badge/SDK-Nexa-blue' alt='Nexa SDK' style='vertical-align: middle;'>
            </a>
        </div>
        """
    )
    gr.HTML(
        """
        <div style="font-family: Arial, sans-serif; font-size: 1em; color: #444;'>
            <b>Powered by Nexa AI SDK🐙</b><br>
        </div>
        """
    )

    prompt_textbox = gr.Textbox(
        placeholder="Describe the audio or summarize key info",
        lines=1,
        label="Prompt Box"
    )

    gr.HTML("<h2>Option 1: Upload Audio File</h2>")
    with gr.Row():
        uploaded_audio = gr.Audio(type="filepath", label="Upload an audio file (wav, mp3)")
        upload_response = gr.Textbox(label="Model Response", interactive=False)
    process_upload_button = gr.Button("Process Uploaded Audio")
    process_upload_button.click(process_audio_fn, inputs=[uploaded_audio, prompt_textbox], outputs=upload_response)

    gr.HTML("<h2>Option 2: Real-time Recording</h2>")
    recorded_audio = gr.Audio(source="microphone", type="filepath", label="Record Audio from Microphone")
    record_response = gr.Textbox(label="Model Response", interactive=False)
    process_record_button = gr.Button("Process Recorded Audio")
    process_record_button.click(process_audio_fn, inputs=[recorded_audio, prompt_textbox], outputs=record_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Model name or path to load")
    parser.add_argument("--local_path", action="store_true", help="Use local path directly")
    parser.add_argument("--huggingface", action="store_true", help="Pull model from Hugging Face")
    parser.add_argument("--modelscope", action="store_true", help="Pull model from ModelScope (not shown in example)")
    parser.add_argument("--local_model_path", type=str, default=None, help="Local model folder or file path")
    parser.add_argument("--projector_local_path", type=str, default=None, help="Path to projector file if needed")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Whether to share the Gradio app publicly")
    args = parser.parse_args()

    
    nexa_model = load_model(
        model_path=args.model_path,
        is_local=args.local_path,
        hf=args.huggingface,
        local_model_path=args.local_model_path,
        projector_local_path=args.projector_local_path
    )

    
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
