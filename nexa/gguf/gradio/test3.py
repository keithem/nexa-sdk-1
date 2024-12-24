import gradio as gr
from nexa.gguf.nexa_inference_voice import NexaVoiceInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model

nexa_model = None
is_local_path = True
hf = False
default_model = "faster-whisper-tiny.en:bin-cpu-fp16"

local_model_path = "/Users/chenzekai/.cache/nexa/hub/official/faster-whisper-tiny.en/bin-cpu-fp16"

def load_model(model_path):

    nexa_model = NexaVoiceInference(
        model_path=model_path,
        local_path=local_model_path
    )
    return nexa_model

try:
    nexa_model = load_model(default_model)
except Exception as e:
    nexa_model = None
    print(f"Failed to load model: {e}")

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
    gr.HTML(
        """
        <div style="display: flex; align-items: center; margin-bottom: 5px; padding-top: 10px;">
            <h1 style="font-family: Arial, sans-serif; font-size: 2.5em; font-weight: bold; margin: 0; padding-bottom: 5px;">
                Nexa AI Voice Transcription
            </h1>
            <a href='https://github.com/NexaAI/nexa-sdk' style='text-decoration: none; margin-left: 10px;'>
                <img src='https://img.shields.io/badge/SDK-Nexa-blue' alt='Nexa SDK' style='vertical-align: middle;'>
            </a>
        </div>
        """
    )
    
    gr.HTML(
        f"""
        <div style="font-family: Arial, sans-serif; font-size: 1em; color: #444;">
            <b>Powered by Nexa AI SDK🐙</b> <br>
            <b>Model path: {default_model}</b>
        </div>
        """
    )

    
    gr.HTML("<h3 style='font-family: Arial, sans-serif; font-size: 1.2em; font-weight: bold;'>Transcription Parameters</h3>")
    with gr.Row():
        beam_size_slider = gr.Slider(
            minimum=1, maximum=10, step=1, value=5, label="Beam Size"
        )
        task_dropdown = gr.Dropdown(
            choices=["transcribe", "translate"],
            value="transcribe",
            label="Task"
        )
        temperature_slider = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Temperature"
        )


    # Option 1: Upload Audio
    gr.HTML("<h2 style='font-family: Arial, sans-serif; font-size: 1.5em; font-weight: bold;'>Option 1: Upload Audio File</h2>")
    with gr.Row():
        uploaded_audio = gr.Audio(type="filepath", label="Upload an audio file (wav, mp3)")
        upload_response = gr.Textbox(label="Model Response", interactive=False)
    process_upload_button = gr.Button("Process Uploaded Audio")
    
    process_upload_button.click(
        process_audio_fn,
        inputs=[uploaded_audio, beam_size_slider, task_dropdown, temperature_slider],
        outputs=upload_response
    )

    # Option 2: Record Audio
    gr.HTML("<h2 style='font-family: Arial, sans-serif; font-size: 1.5em; font-weight: bold;'>Option 2: Real-time Recording</h2>")
    recorded_audio = gr.Audio(source="microphone", type="filepath", label="Record Audio from Microphone")
    record_response = gr.Textbox(label="Model Response", interactive=False)
    process_record_button = gr.Button("Process Recorded Audio")
    process_record_button.click(
        process_audio_fn,
        inputs=[recorded_audio, beam_size_slider, task_dropdown, temperature_slider],
        outputs=record_response
    )

if __name__ == "__main__":
    
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
