import gc
from pathlib import Path

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from huggingface_hub import HfFolder

from nerllama.common.constants import (
    LLAMA_MODELS,
    LLAMA_CHAT_MODELS,
    AUTH_TOKEN_REQUIREMENT_ERROR,
)
from nerllama.fine_tuned.llama2_chat import batch
from nerllama.schemas.DataStruct import create_train_test_instruct_datasets


def generate(model, sources, generation_config):
    model_name = model
    if model is None:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    max_instances = -1
    _, test_dataset = create_train_test_instruct_datasets(
        "../nerllama/data/annotated_nlm.json"
    )
    if max_instances != -1 and max_instances < len(test_dataset):
        test_dataset = test_dataset[:max_instances]

    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []
    input_text = []

    # exit(0)
    # test_dataset = dataset
    for instruction in tqdm(test_dataset):
        target_list.append(instruction["raw_entities"])
        instruction_ids.append(instruction["id"])
        sources.append(instruction["source"])
        input_text.append(instruction["input"])

    target_list = list(batch(target_list, n=1))
    instruction_ids = list(batch(instruction_ids, n=1))
    input_text = list(batch(input_text, n=1))
    for source in tqdm(input_text):
        with torch.no_grad():
            torch.cuda.empty_cache()
            gc.collect()
            messages = [
                {"role": "user", "content": "What is your favourite condiment?"},
                {
                    "role": "assistant",
                    "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
                },
                {"role": "user", "content": "Do you have mayonnaise recipes?"},
            ]

            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

            model_inputs = encodeds.to("cuda")
            model.to("cuda")

            generated_ids = model.generate(
                model_inputs, max_new_tokens=1000, do_sample=True
            )
            decoded = tokenizer.batch_decode(generated_ids)
            print(decoded[0])
        # for s in generation_output.sequences:
        #     string_output = tokenizer.decode(s, skip_special_tokens=True)
        #     extracted_list.append(string_output)
        #     print("Response: ", string_output)


def run(
    model_name=None,
    text=None,
    inputFile=None,
    max_new_tokens=256,
    auth_token=None,
):
    if not model_name or model_name not in LLAMA_CHAT_MODELS:
        raise Exception(f"Invalid model. Model: {model_name}")
    if not auth_token:
        raise Exception(f"Invalid/Empty Auth Token. {AUTH_TOKEN_REQUIREMENT_ERROR}")
    if not inputFile:
        raise Exception(f"Invalid/Empty Dataset file. File: {inputFile}")

    if model_name is None:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    HfFolder.save_token(auth_token)
    try:
        generate(model_name, text, max_new_tokens)
    except:
        generate(model_name, text, max_new_tokens)
