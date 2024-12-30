import gradio as gr
from nexa.gguf.nexa_inference_vlm_omni import NexaOmniVlmInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model
import argparse

nexa_model = None

def load_model(model_path, is_local=False, hf=False, local_model_path=None, projector_local_path=None):
    if is_local:
        local_path = local_model_path
    elif hf:
        local_path, _ = pull_model(model_path, hf=True)
    else:
        local_path, _ = pull_model(model_path)

    nexa_model = NexaOmniVlmInference(
        model_path=model_path,
        local_path=local_path,
        projector_local_path=projector_local_path
    )
    return nexa_model

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

with gr.Blocks() as demo:
    gr.HTML("...title & badges...")

    prompt_textbox = gr.Textbox(
        placeholder="e.g., Describe the image content or ask a question about it.", 
        lines=1, 
        label="Prompt Box"
    )

    gr.HTML("<h2>Upload an image</h2>")
    with gr.Row():
        uploaded_image = gr.Image(type="filepath", label="Upload an image")
        upload_response = gr.Textbox(label="Model Response", interactive=False)
    process_upload_button = gr.Button("Process")
    process_upload_button.click(process_image_fn, inputs=[uploaded_image, prompt_textbox], outputs=upload_response)

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
