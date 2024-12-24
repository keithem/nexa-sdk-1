import gradio as gr
from nexa.gguf.nexa_inference_text import NexaTextInference
from nexa.general import pull_model

# --------------------------------------------------------------------
# 1. 初始化模型相关的变量（保留原先的 is_local_path, hf, nexa_model 等）
# --------------------------------------------------------------------
nexa_model = None
is_local_path = True
hf = False

# --------------------------------------------------------------------
# 2. 默认模型及其本地路径（不做模型选择，保持本地加载）
# --------------------------------------------------------------------
default_model = "Llama3.2-3B-Instruct:q4_0"
local_model_path = "/Users/chenzekai/.cache/nexa/hub/official/Llama3.2-3B-Instruct/q4_0.gguf"

# --------------------------------------------------------------------
# 3. 加载模型函数
# --------------------------------------------------------------------
def load_model(model_path):
    """
    因为 is_local_path=True，这里直接初始化本地模型。
    如果需要更多初始化参数，可在此处添加。
    """
    try:
        # 如果模型已经在本地，不需要再次从远程 pull，则可注释 pull_model
        # local_path, _ = pull_model(model_path)
        nexa_model_instance = NexaTextInference(
            model_path=model_path,
            local_path=local_model_path
        )
        return nexa_model_instance
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

# --------------------------------------------------------------------
# 4. 在启动时加载模型
# --------------------------------------------------------------------
try:
    nexa_model = load_model(default_model)
except Exception as e:
    nexa_model = None
    print(f"Failed to load model: {e}")

# --------------------------------------------------------------------
# 5. 处理文本输入的函数（更新：消费生成器并返回完整文本）
# --------------------------------------------------------------------
def process_text_fn(
    user_input,
    temperature,
    max_new_tokens,
    top_k,
    top_p,
    context_length
):
    """
    将用户文本与生成参数一起传递给模型进行推理，并返回模型输出。
    """
    if not nexa_model:
        return "No model loaded. Please load a model first."
    if not user_input:
        return "Please provide a prompt."

    try:
        # 更新模型内部参数
        nexa_model.params["temperature"] = temperature
        nexa_model.params["max_new_tokens"] = max_new_tokens
        nexa_model.params["top_k"] = top_k
        nexa_model.params["top_p"] = top_p
        nexa_model.params["nctx"] = context_length

        # 原本 _complete() 返回生成器，这里逐个拼接生成完整字符串
        response_iter = nexa_model._complete(user_input)
        response_text = ""
        for chunk in response_iter:
            choice = chunk["choices"][0]
            if "text" in choice:
                token = choice["text"]
            elif "delta" in choice and "content" in choice["delta"]:
                token = choice["delta"]["content"]
            else:
                token = ""
            response_text += token

        return response_text

    except Exception as e:
        return f"Error during text generation: {e}"

# --------------------------------------------------------------------
# 6. Gradio 界面搭建
# --------------------------------------------------------------------
with gr.Blocks() as demo:
    # 标题和徽标
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
    # Powered by + Model path
    gr.HTML(
        f"""
        <div style="font-family: Arial, sans-serif; font-size: 1em; color: #444;">
            <b>Powered by Nexa AI SDK🐙</b> <br>
            <b>Model path: {default_model}</b>
        </div>
        """
    )

    # -----------
    # Generation Parameters
    # -----------
    gr.HTML("<h3 style='font-family: Arial, sans-serif; font-size: 1.2em; font-weight: bold;'>Generation Parameters</h3>")
    with gr.Row():
        temperature_slider = gr.Slider(
            label="Temperature",
            minimum=0.0, maximum=1.0, step=0.01, value=0.7
        )
        max_new_tokens_slider = gr.Slider(
            label="Max New Tokens",
            minimum=1, maximum=500, step=1, value=128
        )
    with gr.Row():
        top_k_slider = gr.Slider(
            label="Top K",
            minimum=1, maximum=100, step=1, value=40
        )
        top_p_slider = gr.Slider(
            label="Top P",
            minimum=0.0, maximum=1.0, step=0.01, value=0.9
        )
    with gr.Row():
        context_length_slider = gr.Slider(
            label="Context length",
            minimum=1000, maximum=9999, step=1, value=2048
        )

    # ----------
    # 文本输入
    # ----------
    user_input_box = gr.Textbox(
        placeholder="Say something...",
        lines=3,
        show_label=False
    )

    # -----------
    # 触发按钮 + 输出
    # -----------
    send_button = gr.Button("Send")
    model_response = gr.Textbox(
        label="Model Response",
        interactive=False
    )

    send_button.click(
        fn=process_text_fn,
        inputs=[
            user_input_box,
            temperature_slider,
            max_new_tokens_slider,
            top_k_slider,
            top_p_slider,
            context_length_slider
        ],
        outputs=model_response
    )

# --------------------------------------------------------------------
# 7. 启动 Gradio 应用
# --------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
