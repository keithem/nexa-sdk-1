import gradio as gr
from nexa.gguf.nexa_inference_text import NexaTextInference
from nexa.gguf.llama._utils_transformers import suppress_stdout_stderr
from nexa.general import pull_model

# -----------------------------
# 1) 初始化模型相关变量（与需求保持一致：默认本地路径，无模型选择）
# -----------------------------
nexa_model = None
is_local_path = True
hf = False
default_model = "Llama3.2-3B-Instruct:q4_0"  # 第2条：修改为此默认值

# 这里指定模型在本地的实际文件路径
local_model_path = "/Users/chenzekai/.cache/nexa/hub/official/Llama3.2-3B-Instruct/q4_0.gguf"

# -----------------------------
# 加载模型函数
# -----------------------------
def load_model(model_path):
    """
    加载文本模型。由于 is_local_path=True, 我们将直接加载本地路径。
    """
    nexa_model = NexaTextInference(
        model_path=model_path,
        local_path=local_model_path,
        # 可按需添加更多初始化参数，如: temperature, max_new_tokens 等
    )
    return nexa_model

# 尝试加载模型
try:
    nexa_model = load_model(default_model)
except Exception as e:
    nexa_model = None
    print(f"Failed to load model: {e}")

# -----------------------------
# 推理函数：根据用户输入（文本）和超参数，返回生成结果
# -----------------------------
def process_text_fn(user_input, temperature, max_new_tokens, top_k, top_p, nctx):
    """
    将文本输入和模型参数传给 NexaTextInference 进行文本生成，并返回结果。
    """
    if not nexa_model:
        return "No model loaded. Please check the local path or model loading."

    # 先更新模型内部使用的生成参数
    nexa_model.params.update({
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "nctx": nctx,
    })

    try:
        # 如果模型是 chat_format 的，则用 _chat，否则用 _complete
        if hasattr(nexa_model, "chat_format") and nexa_model.chat_format:
            # 流式输出
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
            # 普通文本生成
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

# -----------------------------
# 构建 Gradio 界面
# -----------------------------
with gr.Blocks() as demo:
    # 第3条：修改标题为“Nexa AI Text Generation”
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

    # 第5条：Generation Parameters 板块
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
                value=0.7  # 可根据需要默认设一个
            )
            max_tokens_slider = gr.Slider(
                label="Max New Tokens",
                minimum=1,
                maximum=500,
                step=1,
                value=256  # 可根据需要默认设一个
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

    # 第4条：只保留一个文本输入框, placeholder="Say something"
    user_text_input = gr.Textbox(
        placeholder="Say something",
        lines=3,
        label="",  # 需求里提到：去掉 label
    )

    # 第6条：点击 Send 后输出到 “Model Response”
    send_button = gr.Button("Send")
    model_response = gr.Textbox(label="Model Response", interactive=False)

    # Send 按钮点击后执行
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

# -----------------------------
# 启动 Gradio 服务
# -----------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
