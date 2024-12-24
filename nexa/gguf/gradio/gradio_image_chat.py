import gradio as gr
from nexa.gguf.nexa_inference_image import NexaImageInference

model_file_path = "/Users/chenzekai/.cache/nexa/hub/official/lcm-dreamshaper-v7/fp16.gguf"

default_model = "lcm-dreamshaper-v7:fp16"

nexa_model = None

def load_model(model_path):

    nexa_model = NexaImageInference(
        model_path=model_path,
        local_path=model_path
    )
    return nexa_model

try:
    nexa_model = load_model(model_file_path)
except Exception as e:
    nexa_model = None
    print(f"Failed to load model: {e}")

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
    
    gr.HTML(
        """
        <div style="display: flex; align-items: center; margin-bottom: 5px; padding-top: 10px;">
            <h1 style="font-family: Arial, sans-serif; font-size: 2.5em; font-weight: bold; margin: 0; padding-bottom: 5px;">
                Nexa AI Image Generation
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

    # Generation Parameters
    gr.Markdown("## Generation Parameters")
    with gr.Row():
        steps_slider = gr.Slider(
            minimum=1, maximum=8, step=1, value=4, label="Inference Steps"
        )
        height_slider = gr.Slider(
            minimum=64, maximum=1024, step=1, value=512, label="Height"
        )
        width_slider = gr.Slider(
            minimum=64, maximum=1024, step=1, value=512, label="Width"
        )
        guidance_slider = gr.Slider(
            minimum=0.0, maximum=2.0, step=0.01, value=1.0, label="Guidance Scale"
        )
        seed_slider = gr.Slider(
            minimum=0, maximum=10000, step=1, value=42, label="Random Seed"
        )

    gr.Markdown("## Enter your prompt:")
    prompt_textbox = gr.Textbox(
        placeholder="e.g., A majestic lion wearing a crown", 
        lines=1, 
        label="Prompt"
    )

    gr.Markdown("## Enter your negative prompt (optional):")
    negative_prompt_textbox = gr.Textbox(
        placeholder="e.g., Overexposure, blur, disfigured", 
        lines=1, 
        label="Negative Prompt"
    )

    output_image = gr.Image(label="Generated Image")
    output_message = gr.Textbox(label="Model Message", interactive=False)

    generate_button = gr.Button("Generate Image")

    generate_button.click(
        fn=generate_images_fn,
        inputs=[
            steps_slider,
            height_slider,
            width_slider,
            guidance_slider,
            seed_slider,
            prompt_textbox,
            negative_prompt_textbox
        ],
        outputs=[output_image, output_message]
    )

    gr.HTML(
        """
        <p style='font-family: Arial, sans-serif;'>
            <b>How to save:</b> Right-click on the generated image above and select "Save Image As..." to download.
        </p>
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
