import gradio as gr
from nexa.gguf.nexa_inference_text import NexaTextInference
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

    return NexaTextInference(
        model_path=model_path,
        local_path=local_path,
    )

def process_text_fn(user_input, temperature, max_new_tokens, top_k, top_p, nctx):
    if not nexa_model:
        return "No model loaded. Please check the local path or model loading."

    nexa_model.params.update({
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "nctx": nctx,
    })

    try:
        result = ""
        with suppress_stdout_stderr():
            if hasattr(nexa_model, "chat_format") and nexa_model.chat_format:
                for chunk in nexa_model._chat(user_input):
                    choice = chunk["choices"][0]
                    if "delta" in choice:
                        delta = choice["delta"]
                        content = delta.get("content", "")
                    else:
                        content = choice.get("text", "")
                    result += content
            else:
                for chunk in nexa_model._complete(user_input):
                    choice = chunk["choices"][0]
                    if "text" in choice:
                        delta = choice["text"]
                    elif "delta" in choice:
                        delta = choice["delta"]["content"]
                    else:
                        delta = ""
                    result += delta
        return result
    except Exception as e:
        return f"Error during text generation: {e}"

with gr.Blocks() as demo:
    gr.HTML("...title & badges...")

    with gr.Group():
        gr.HTML("<h2>Generation Parameters</h2>")
        with gr.Row():
            temperature_slider = gr.Slider(label="Temperature", minimum=0.00, maximum=1.00, step=0.01, value=0.7)
            max_tokens_slider = gr.Slider(label="Max New Tokens", minimum=1, maximum=500, step=1, value=256)
        with gr.Row():
            top_k_slider = gr.Slider(label="Top K", minimum=1, maximum=100, step=1, value=50)
            top_p_slider = gr.Slider(label="Top P", minimum=0.00, maximum=1.00, step=0.01, value=1.0)
        with gr.Row():
            nctx_slider = gr.Slider(label="Context length", minimum=1000, maximum=9999, step=1, value=2048)

    user_text_input = gr.Textbox(placeholder="Say something", lines=3, label="")
    send_button = gr.Button("Send")
    model_response = gr.Textbox(label="Model Response", interactive=False)

    send_button.click(
        fn=process_text_fn,
        inputs=[user_text_input, temperature_slider, max_tokens_slider, top_k_slider, top_p_slider, nctx_slider],
        outputs=model_response
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
