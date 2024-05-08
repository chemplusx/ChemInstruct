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
    AUTH_TOKEN_REQUIREMENT_ERROR,
)
from nerllama.schemas.DataStruct import INSTRUCTION_TEXT

def generate(model, source_text, generation_config):
    if not model:
        model_name = "tiiuae/falcon-7b-instruct"
    else:
        model_name = model

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    user_query = "Extract all chemical entities from the provided text below. \nText: `{}`"

    device = "mps" if torch.backends.mps.is_built() else ('cuda' if torch.cuda.is_available() else 'cpu')
    torch_dtype = torch.float16 if device != 'cuda' else torch.bfloat16

    with torch.no_grad():
        torch.cuda.empty_cache()
        # gc.collect()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="mps",
        )

        messages = [
            {"role": "system", "content": INSTRUCTION_TEXT},
            {"role": "user", "content": user_query.format(source_text)},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                # add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id
        ]

        generation_output = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.2,
        )
    print("Response: ", generation_output[0]["generated_text"])


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

    content = ""
    if text:
        content = text
    else:
        try:
            content += inputFile.read()
        except Exception as ex:
            raise Exception("Unable to read from file" + inputFile)

    try:
        with wandb.init(project="Instruction NER") as run:
            generate(model_name, content, max_new_tokens)
    except:
        generate(model_name, content, max_new_tokens)
