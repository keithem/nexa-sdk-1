import gradio as gr
from nexa.gguf.nexa_inference_vlm_omni import NexaOmniVlmInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model

# Initialize model variables
nexa_model = None
is_local_path = True
hf = False

# Set the local model and projector paths here
local_model_path = "/Users/chenzekai/.cache/nexa/hub/official/omniVLM/model-fp16.gguf"
projector_local_path = "/Users/chenzekai/.cache/nexa/hub/official/omniVLM/projector-fp16.gguf"

default_model = "omniVLM:fp16"

def load_model(model_path):
    nexa_model = NexaOmniVlmInference(
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

def process_image_fn(image_file, prompt=""):
    if not nexa_model:
        return "No model loaded. Please load a model first."
    if not image_file:
        return "Please provide an image input."
    try:
        with suppress_stdout_stderr():
            response = nexa_model.inference(prompt, image_file)
        return response
    except Exception as e:
        return f"Error during image processing: {e}"

# Gradio interface
with gr.Blocks() as demo:
    # Title and badges in the same line
    gr.HTML(
        """
        <div style="display: flex; align-items: center; margin-bottom: 5px; padding-top: 10px;">
            <h1 style="font-family: Arial, sans-serif; font-size: 2.5em; font-weight: bold; margin: 0; padding-bottom: 5px;">
                Nexa AI Omni VLM Generation
            </h1>
            <a href='https://github.com/NexaAI/nexa-sdk' style='text-decoration: none; margin-left: 10px;'>
                <img src='https://img.shields.io/badge/SDK-Nexa-blue' alt='Nexa SDK' style='vertical-align: middle;'>
            </a>
        </div>
        """
    )
    # Powered by and Model Path
    gr.HTML(
        f"""
        <div style="font-family: Arial, sans-serif; font-size: 1em; color: #444;">
            <b>Powered by Nexa AI SDK🐙</b> <br>
            <b>Model path: {default_model}</b>
        </div>
        """
    )

    # Enter your text input
    gr.HTML("<h3 style='font-family: Arial, sans-serif; font-size: 1.1em; font-weight: bold;'>Enter your text input:</h3>")
    prompt_textbox = gr.Textbox(
        placeholder="e.g., Describe the image content or ask a question about it.", 
        lines=1, 
        label="Prompt Box"
    )

    #Upload an image
    gr.HTML("<h2 style='font-family: Arial, sans-serif; font-size: 1.5em; font-weight: bold;'>Upload an image</h2>")
    with gr.Row():
        uploaded_image = gr.Image(type="filepath", label="Upload an image (png, jpg, jpeg)")
        upload_response = gr.Textbox(label="Model Response", interactive=False)
    process_upload_button = gr.Button("Process")
    process_upload_button.click(
        process_image_fn, 
        inputs=[uploaded_image, prompt_textbox], 
        outputs=upload_response
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
