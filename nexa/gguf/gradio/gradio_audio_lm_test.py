import gradio as gr
from nexa.gguf.nexa_inference_audio_lm import NexaAudioLMInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model

# Initialize model variables
nexa_model = None
is_local_path = True
hf = False
# Set the local model and projector paths here
local_model_path = "/Users/chenzekai/.cache/nexa/hub/official/OmniAudio-2.6B/model-q4_K_M.gguf"
projector_local_path = "/Users/chenzekai/.cache/nexa/hub/official/OmniAudio-2.6B/projector-q4_K_M.gguf"

default_model = "omniaudio"  # Keep this as 'omniaudio' for display purposes

def load_model(model_path):
    # Since is_local_path=True, and we know the paths, we directly initialize
    nexa_model = NexaAudioLMInference(
        model_path=model_path,
        local_path=local_model_path,
        projector_local_path=projector_local_path
    )
    return nexa_model

try:
    nexa_model = load_model(default_model)
except Exception as e:
    nexa_model = None
    print(f"Failed to load model: {e}")

def process_audio_fn(audio_file, prompt=""):
    if not nexa_model:
        return "No model loaded. Please load a model first."
    if not audio_file:
        return "Please provide an audio input."
    
    try:
        with suppress_stdout_stderr():
            # Use the 'inference' method for OmniAudio
            response = nexa_model.inference(audio_file, prompt)
        return response
    except Exception as e:
        return f"Error during audio processing: {e}"

# Gradio interface
with gr.Blocks() as demo:
    # Title and badges in the same line
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
    # Powered by and Model Path
    # Keep the UI text unchanged, do not show actual paths
    gr.HTML(
        """
        <div style="font-family: Arial, sans-serif; font-size: 1em; color: #444;">
            <b>Powered by Nexa AI SDK🐙</b> <br>
            <b>Model path: omniaudio</b>
        </div>
        """
    )

    # Optional Prompt
    gr.HTML("<h3 style='font-family: Arial, sans-serif; font-size: 1.1em; font-weight: bold;'>Enter optional prompt text:</h3>")
    prompt_textbox = gr.Textbox(placeholder="e.g., Describe the audio content or summarize key information.", lines=1, label="Prompt Box")

    # Option 1: Upload Audio File
    gr.HTML("<h2 style='font-family: Arial, sans-serif; font-size: 1.5em; font-weight: bold;'>Option 1: Upload Audio File</h2>")
    with gr.Row():
        uploaded_audio = gr.Audio(type="filepath", label="Upload an audio file (wav, mp3)")
        upload_response = gr.Textbox(label="Model Response", interactive=False)
    process_upload_button = gr.Button("Process Uploaded Audio")
    process_upload_button.click(process_audio_fn, inputs=[uploaded_audio, prompt_textbox], outputs=upload_response)

    # Option 2: Record Audio
    gr.HTML("<h2 style='font-family: Arial, sans-serif; font-size: 1.5em; font-weight: bold;'>Option 2: Real-time Recording</h2>")
    recorded_audio = gr.Audio(source="microphone", type="filepath", label="Record Audio from Microphone")
    record_response = gr.Textbox(label="Model Response", interactive=False)
    process_record_button = gr.Button("Process Recorded Audio")
    process_record_button.click(process_audio_fn, inputs=[recorded_audio, prompt_textbox], outputs=record_response)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
