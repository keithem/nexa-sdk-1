import gradio as gr
from nexa.gguf.nexa_inference_text import NexaTextInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model

nexa_model = None
is_local_path = True
hf = False
default_model = "Llama3.2-3B-Instruct:q4_0"  

local_model_path = "/Users/chenzekai/.cache/nexa/hub/official/Llama3.2-3B-Instruct/q4_0.gguf"

def load_model(model_path):

    nexa_model = NexaTextInference(
        model_path=model_path,
        local_path=local_model_path,
    )
    return nexa_model


try:
    nexa_model = load_model(default_model)
except Exception as e:
    nexa_model = None
    print(f"Failed to load model: {e}")


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
        
        if hasattr(nexa_model, "chat_format") and nexa_model.chat_format:
            
            result = ""
            with suppress_stdout_stderr():
                for chunk in nexa_model._chat(user_input):
                    choice = chunk["choices"][0]
                    if "delta" in choice:
                        delta = choice["delta"]
                        content = delta.get("content", "")
                    else:
                        content = choice.get("text", "")
                    result += content
            return result
        else:
            
            result = ""
            with suppress_stdout_stderr():
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
    
    gr.HTML(
        """
        <div style="display: flex; align-items: center; margin-bottom: 5px; padding-top: 10px;">
            <h1 style="font-family: Arial, sans-serif; font-size: 2.5em; font-weight: bold; margin: 0; padding-bottom: 5px;">
                Nexa AI Text Generation
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

    
    with gr.Group():
        gr.HTML(
            """
            <h2 style='font-family: Arial, sans-serif; font-size: 1.3em; font-weight: bold; margin-bottom: 0.2em;'>
                Generation Parameters
            </h2>
            """
        )
        with gr.Row():
            temperature_slider = gr.Slider(
                label="Temperature",
                minimum=0.00,
                maximum=1.00,
                step=0.01,
                value=0.7  
            )
            max_tokens_slider = gr.Slider(
                label="Max New Tokens",
                minimum=1,
                maximum=500,
                step=1,
                value=256 
            )
        with gr.Row():
            top_k_slider = gr.Slider(
                label="Top K",
                minimum=1,
                maximum=100,
                step=1,
                value=50
            )
            top_p_slider = gr.Slider(
                label="Top P",
                minimum=0.00,
                maximum=1.00,
                step=0.01,
                value=1.0
            )
        with gr.Row():
            nctx_slider = gr.Slider(
                label="Context length",
                minimum=1000,
                maximum=9999,
                step=1,
                value=2048
            )


    user_text_input = gr.Textbox(
        placeholder="Say something",
        lines=3,
        label="",  
    )


    send_button = gr.Button("Send")
    model_response = gr.Textbox(label="Model Response", interactive=False)


    send_button.click(
        fn=process_text_fn,
        inputs=[
            user_text_input,
            temperature_slider,
            max_tokens_slider,
            top_k_slider,
            top_p_slider,
            nctx_slider
        ],
        outputs=model_response
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)