import gc
from pathlib import Path
import os

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from huggingface_hub.hf_api import HfFolder

from nerllama.common.constants import (
    LLAMA_MODELS,
    LLAMA_CHAT_MODELS,
    AUTH_TOKEN_REQUIREMENT_ERROR,
)
from nerllama.fine_tuned.llama2_chat import batch
from nerllama.schemas.DataStruct import create_train_test_instruct_datasets

def generate(model, source_text, generation_config):
    if not model:
        model_name = "tiiuae/falcon-7b-instruct"
    else:
        model_name = model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    instruction = "Extract all chemical entities from the provided text. Text: "

    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        generation_output = pipeline(
            instruction + source_text,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
    for s in generation_output:
        print("Response: ", s)


def run(
    model_name=None,
    text=None,
    inputFile=None,
    max_new_tokens=256,
    auth_token=None,
):
    if not model_name:
        raise Exception(f"Invalid model. Model: {model_name}")
    if not auth_token:
        raise Exception(f"Invalid/Empty Auth Token. {AUTH_TOKEN_REQUIREMENT_ERROR}")
    if not inputFile:
        raise Exception(f"Invalid/Empty Dataset file. File: {inputFile}")

    if model_name is None:
        model_name = "tiiuae/falcon-7b-instruct"

    HfFolder.save_token(auth_token)
    try:
        with wandb.init(project="Instruction NER") as run:
            generate(model_name, text, max_new_tokens)
    except:
        generate(model_name, text, max_new_tokens)
