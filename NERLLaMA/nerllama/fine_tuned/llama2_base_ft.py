from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
import os
import torch
from transformers import (
    LlamaTokenizer,
)

from huggingface_hub.hf_api import HfFolder

from nerllama.common.constants import LLAMA_MODELS, AUTH_TOKEN_REQUIREMENT_ERROR
from nerllama.schemas.Conversation import preprocess_instance

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

HfFolder.save_token("<your_hf_api_token>")


def get_response(responses):
    responses = [r.split("ASSISTANT:")[-1].strip() for r in responses]
    return responses


def generate(model, text, max_new_tokens=256):
    entity_type = "chemical"
    conversation = [
        {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"},
                {"from": "gpt", "value": "I've read this text."},
                {
                    "from": "human",
                    "value": f"What describes {entity_type} in the text?",
                },
                {"from": "gpt", "value": "[]"},
            ]
        }
    ]
    prompts = [preprocess_instance(convo["conversations"]) for convo in conversation]
    sampling_params = SamplingParams(
        temperature=0.1, max_tokens=max_new_tokens, stop=["</s>"]
    )
    responses = model.generate(prompts, sampling_params)
    responses_corret_order = []
    response_set = {response.prompt: response for response in responses}
    for prompt in prompts:
        assert prompt in response_set
        responses_corret_order.append(response_set[prompt])
    responses = responses_corret_order
    outputs = get_response([output.outputs[0].text for output in responses])
    return outputs


def run(
    model_name=None,
    text=None,
    inputFile=None,
    max_new_tokens=256,
    auth_token=None,
):
    if not model_name or model_name not in LLAMA_MODELS:
        raise Exception(f"Invalid model. Model: {model_name}")
    if not auth_token:
        raise Exception(f"Invalid/Empty Auth Token. {AUTH_TOKEN_REQUIREMENT_ERROR}")
    if (not inputFile or not Path(inputFile).exists()) and not text:
        raise Exception(f"Invalid/Empty Dataset file. File: {inputFile}")

    if model_name is None:
        model_name = "ChemPlusX/llama2-base-ft-NER"

    model = LLM(model=model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    if text is not None:
        print("Response: ", generate(model, text, max_new_tokens=max_new_tokens))
    elif inputFile is not None:
        data = pd.read_csv(inputFile, encoding="cp1252")
        inputs = data.get("text")
        for i in range(len(inputs)):
            print("Response: ", generate(inputs[i], max_new_tokens=max_new_tokens))
    else:
        raise ValueError("Either text or inputFile must be provided.")
