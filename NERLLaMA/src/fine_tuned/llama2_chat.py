import gc
from pathlib import Path

import torch
import wandb
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    GenerationConfig,
)
from peft import PeftConfig, PeftModel
from huggingface_hub.hf_api import HfFolder

from NERLLaMA.src.common.constants import LLAMA_MODELS, AUTH_TOKEN_REQUIREMENT_ERROR
from NERLLaMA.src.schemas.DataStruct import create_train_test_instruct_datasets

HfFolder.save_token("<your_hf_api_token>")

token = "<your_hf_api_token>"


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        try:
            t = iterable[ndx : min(ndx + n, l)]
        except Exception:
            print("Error")
        yield t


def generate(model, sources, generation_config):
    model_name = "chrohi/llama2-NLMC-NER-FT"

    generation_config = GenerationConfig.from_pretrained(model_name, token=token)

    peft_config = PeftConfig.from_pretrained(model_name, token=token)
    base_model_name = peft_config.base_model_name_or_path

    models = {
        "llama": AutoModelForCausalLM,
        "t5": T5ForConditionalGeneration,
        "mistral": AutoModelForCausalLM,
    }

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, load_in_8bit=True, device_map="auto", token=token
    )

    model = PeftModel.from_pretrained(model, model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    model.eval()
    model = torch.compile(model)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    max_instances = -1
    _, test_dataset = create_train_test_instruct_datasets(
        "../src/data/annotated_nlm.json"
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
        input_ids = tokenizer(source, return_tensors="pt", padding=True)[
            "input_ids"
        ].cuda()
        with torch.no_grad():
            torch.cuda.empty_cache()
            gc.collect()
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
            )
        for s in generation_output.sequences:
            string_output = tokenizer.decode(s, skip_special_tokens=True)
            extracted_list.append(string_output)
            print("Response: ", string_output)


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
    if not inputFile or not Path(inputFile).exists():
        raise Exception(f"Invalid/Empty Dataset file. File: {inputFile}")

    if model_name is None:
        model_name = "ChemPlusX/llama2-base-ft-NER"

    try:
        with wandb.init(project="Instruction NER") as run:
            generate(model_name, text, max_new_tokens)
    except:
        generate(model_name, text, max_new_tokens)
