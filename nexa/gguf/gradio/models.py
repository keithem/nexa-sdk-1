from nexa.general import pull_model
from nexa.gguf.nexa_inference_text import NexaTextInference
from nexa.gguf.nexa_inference_image import NexaImageInference
from nexa.gguf.nexa_inference_vlm import NexaVLMInference
from nexa.gguf.nexa_inference_voice import NexaVoiceInference
from nexa.gguf.nexa_inference_audio_lm import NexaAudioLMInference

default_text_model = "your_text_model_name"
default_image_model = "your_image_model_name"
default_multimodal_model = "your_multimodal_model_name"
default_audio_model = "your_audio_model_name"
default_audio_lm_model = "your_audio_lm_model_name"

def load_text_model(model_path):
    local_path, _ = pull_model(model_path)
    return NexaTextInference(model_path=model_path, local_path=local_path)

def load_image_model(model_path):
    local_path, _ = pull_model(model_path)
    return NexaImageInference(model_path=model_path, local_path=local_path)

def load_multimodal_model(model_path):
    local_path, _ = pull_model(model_path)
    return NexaVLMInference(model_path=model_path, local_path=local_path)

def load_audio_model(model_path):
    local_path, _ = pull_model(model_path)
    return NexaVoiceInference(model_path=model_path, local_path=local_path)

def load_audio_lm_model(model_path):
    local_path, _ = pull_model(model_path)
    return NexaAudioLMInference(model_path=model_path, local_path=local_path)

text_model = load_text_model(default_text_model)
image_model = load_image_model(default_image_model)
multimodal_model = load_multimodal_model(default_multimodal_model)
audio_model = load_audio_model(default_audio_model)
audio_lm_model = load_audio_lm_model(default_audio_lm_model)
