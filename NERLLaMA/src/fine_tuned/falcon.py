import gc
from pathlib import Path

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    # model_name = "vilsonrodrigues/falcon-7b-instruct-sharded"

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
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=False,
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             #device_map = device_map,
    # quantization_config=quantization_config, 
    trust_remote_code=True)

    generation_config = model.generation_config

    generation_config.temperature = 0
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = 256
    generation_config.use_cache=True
    generation_config.eos_token_id=tokenizer.eos_token_id
    generation_config.pad_token_id=tokenizer.eos_token_id

    # pipeline = transformers.pipeline(
    #             "text-generation",
    #             model=model,
    #             tokenizer=tokenizer,
    #             #torch_dtype=torch.bfloat16,
    #             trust_remote_code=True,
    #             device_map="auto",
    #             #device_map=device_map,
    #             max_length=200,
    #             use_cache=True,
    #             num_return_sequences=1,
    #             eos_token_id=tokenizer.eos_token_id,
    #             pad_token_id=tokenizer.eos_token_id,
    #         )
    # generation_output = pipeline(
    #             "Create a list of four important things a pilot should take note of"
    #         )
    # for s in generation_output:
    #     print("Response: ", s)
    
    # return
    
    for instruction in tqdm(test_dataset):
        target_list.append(instruction["raw_entities"])
        instruction_ids.append(instruction["id"])
        sources.append(instruction["input"])

    target_list = list(batch(target_list, n=1))
    instruction_ids = list(batch(instruction_ids, n=1))
    sources = list(batch(sources, n=1))
    for source in tqdm(sources):
        print(source)
        prompt = """
            Human: Extract all Chemical entities from the given text. Text The N-containing portions of milk can be divided into three broad fractions, including casein nitrogen (CN), whey protein nitrogen (WPN), and non protein nitrogen (NPN).
            AI:
        """
        inputids = tokenizer(prompt, return_tensors='pt').input_ids
        inputids = inputids.to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids = inputids,
                generation_config = generation_config,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        # with torch.no_grad():
        #     torch.cuda.empty_cache()
        #     gc.collect()
            
        #     # generation_output = pipeline(
        #     #     "Create a list of four important things a pilot should take note of"
        #     # )

            
        # for s in generation_output:
        #     print("Response: ", s)


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
