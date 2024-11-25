import sys
import tempfile
import gradio as gr
from PIL import Image
from nexa.general import pull_model
from nexa.gguf.nexa_inference_vlm_omni import NexaOmniVlmInference

default_model = sys.argv[1]
is_local_path = False if sys.argv[2] == "False" else True
hf = False if sys.argv[3] == "False" else True
projector_local_path = sys.argv[4] if len(sys.argv) > 4 else None

# Load model once at startup instead of using cache
def load_model(model_path):
    if is_local_path:
        local_path = model_path
    elif hf:
        local_path, _ = pull_model(model_path, hf=True)
    else:
        local_path, _ = pull_model(model_path)
        
    if is_local_path:
        nexa_model = NexaOmniVlmInference(model_path=model_path, local_path=local_path, projector_local_path=projector_local_path)
    else:
        nexa_model = NexaOmniVlmInference(model_path=model_path, local_path=local_path)
    return nexa_model

nexa_model = load_model(default_model)

def generate_response(image, text):
    if image is None:
        return "Please upload an image"
    if not text.strip():
        return "Please enter some text"
    
    # Save uploaded image to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
        Image.fromarray(image).save(temp_file.name)
        response = nexa_model.inference(text, temp_file.name)
    return response

# Create Gradio interface
demo = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Image(label="Upload Image"),
        gr.Textbox(label="Enter your text input", placeholder="Type your prompt here...")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Nexa AI Omni VLM Generation",
    description="Powered by Nexa AI SDK🐙",
)

if __name__ == "__main__":
    demo.launch()