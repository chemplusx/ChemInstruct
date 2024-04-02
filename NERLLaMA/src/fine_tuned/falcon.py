import gc
from pathlib import Path

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from NERLLaMA.src.common.constants import (
    LLAMA_MODELS,
    LLAMA_CHAT_MODELS,
    AUTH_TOKEN_REQUIREMENT_ERROR,
)
from NERLLaMA.src.fine_tuned.llama2_chat import batch
from NERLLaMA.src.schemas.DataStruct import create_train_test_instruct_datasets


def generate(model, sources, generation_config):
    model_name = "tiiuae/falcon-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    max_instances = -1
    _, test_dataset = create_train_test_instruct_datasets(
        "/mnt/d/workspace/ChemInstruct/NERLLaMA/src/data/annotated_nlm.json"
    )
    if max_instances != -1 and max_instances < len(test_dataset):
        test_dataset = test_dataset[:max_instances]

    extracted_list = []
    target_list = []
    instruction_ids = []
    sources = []

    # exit(0)
    # test_dataset = dataset
    for instruction in tqdm(test_dataset):
        target_list.append(instruction["raw_entities"])
        instruction_ids.append(instruction["id"])
        sources.append(instruction["source"])

    target_list = list(batch(target_list, n=1))
    instruction_ids = list(batch(instruction_ids, n=1))
    sources = list(batch(sources, n=1))
    for source in tqdm(sources):
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
                source,
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
    if not model_name or model_name not in LLAMA_CHAT_MODELS:
        raise Exception(f"Invalid model. Model: {model_name}")
    if not auth_token:
        raise Exception(f"Invalid/Empty Auth Token. {AUTH_TOKEN_REQUIREMENT_ERROR}")
    if not inputFile or not Path(inputFile).exists():
        raise Exception(f"Invalid/Empty Dataset file. File: {inputFile}")

    if model_name is None:
        model_name = "tiiuae/falcon-7b-instruct"

    try:
        with wandb.init(project="Instruction NER") as run:
            generate(model_name, text, max_new_tokens)
    except:
        generate(model_name, text, max_new_tokens)
