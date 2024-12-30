import gradio as gr
from nexa.gguf.nexa_inference_image import NexaImageInference
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

    return NexaImageInference(
        model_path=model_path,
        local_path=local_path
    )

def generate_images_fn(steps, height, width, guidance, seed, prompt, negative_prompt):
    if not nexa_model:
        return [None, "No model loaded. Please check your model path or environment."]
    if not prompt:
        return [None, "Please enter a prompt to proceed."]

    nexa_model.params.update({
        "num_inference_steps": steps,
        "height": height,
        "width": width,
        "guidance_scale": guidance,
        "random_seed": seed,
    })

    try:
        images = nexa_model.txt2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=nexa_model.params["guidance_scale"],
            width=nexa_model.params["width"],
            height=nexa_model.params["height"],
            sample_steps=nexa_model.params["num_inference_steps"],
            seed=nexa_model.params["random_seed"]
        )
        if len(images) > 0:
            return [images[0], "Image generated successfully!"]
        else:
            return [None, "No image generated. Check your parameters or prompt."]
    except Exception as e:
        return [None, f"Error during image generation: {e}"]

with gr.Blocks() as demo:
    gr.HTML("...header...")

    with gr.Row():
        steps_slider = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="Inference Steps")
        height_slider = gr.Slider(minimum=64, maximum=1024, step=1, value=512, label="Height")
        width_slider = gr.Slider(minimum=64, maximum=1024, step=1, value=512, label="Width")
        guidance_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=1.0, label="Guidance Scale")
        seed_slider = gr.Slider(minimum=0, maximum=10000, step=1, value=42, label="Random Seed")

    prompt_textbox = gr.Textbox(placeholder="e.g., A majestic lion...", lines=1, label="Prompt")
    negative_prompt_textbox = gr.Textbox(placeholder="e.g., Overexposed, blur...", lines=1, label="Negative Prompt")

    output_image = gr.Image(label="Generated Image")
    output_message = gr.Textbox(label="Model Message", interactive=False)

    generate_button = gr.Button("Generate Image")
    generate_button.click(
        fn=generate_images_fn,
        inputs=[steps_slider, height_slider, width_slider, guidance_slider, seed_slider, prompt_textbox, negative_prompt_textbox],
        outputs=[output_image, output_message]
    )

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
