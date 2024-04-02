LLAMA_MODELS = [
    "ChemPlusX/llama2-chat-ft-NER",
    "ChemPlusX/llama2-base-ft-NER",
    "meta-llama/Llama-2-7b-chat-hf",
]
LLAMA_MODEL_MAP = {
    "llama2-chat-ft": "ChemPlusX/llama2-chat-ft-NER",
    "llama2-base-ft": "ChemPlusX/llama2-base-ft-NER",
    "llama2-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-chat-70b": "meta-llama/Llama-2-70b-chat-hf",
    "mistral-chat-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "falcon-chat-7b": "tiiuae/falcon-7b-instruct",
    "falcon-chat-180b": "tiiuae/falcon-180B-chat",
}
LLAMA_CHAT_MODELS = [
    "ChemPlusX/llama-chat-ft-NER",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-180B-chat",
]


HF_MODEL_ACCESS = "https://huggingface.co/meta-llama/Llama-2-7b"
AUTH_TOKEN_REQUIREMENT_ERROR = f"HuggingFace Auth token is required to authenticate the use of LLaMA2 models. Access to LLaMA models can be requested here <{HF_MODEL_ACCESS}>"
