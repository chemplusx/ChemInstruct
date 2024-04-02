from functools import cache
from random import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import TypedDict
from NERLLaMA.src.schemas.Conversation import Conversation, get_conv_template

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PreTrainedTokenizer,
)

from NERLLaMA.src.schemas.Misc import tokenize
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class ChemStruct(TypedDict):
    instruction: str
    input: str
    output: str
    source: str
    raw_entities: dict[str, list[str]]
    id: str


class InstructDataset(Dataset):
    def __init__(
        self,
        instructions: list[ChemStruct],
        tokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        model_type: str = "llama",
        only_target_loss: bool = True,
        padding: bool = False,
    ):
        self.instructions = instructions
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.model_type = model_type
        self.only_target_loss = only_target_loss
        self.padding = padding

        self.processed_instructions = []

        for instruction in tqdm(self.instructions):
            if self.model_type in ["llama", "mistral"]:
                tensors = self.convert_instruction_causal(instruction)
            else:
                raise ValueError('model_type must be equals "llama", "mistral"')

            self.processed_instructions.append(tensors)

    def __len__(self):
        return len(self.processed_instructions)

    def __getitem__(self, index):
        return self.processed_instructions[index]

    def convert_instruction_causal(self, instruction: dict[str, str]):
        target = instruction["output"]
        source = instruction["source"]

        source_tokens = self.tokenizer(
            source,
            add_special_tokens=False,
            max_length=self.max_source_tokens_count,
            padding=False,
            truncation=True,
        )["input_ids"]

        if self.tokenizer.bos_token_id:
            source_tokens.insert(0, self.tokenizer.bos_token_id)

        input_ids = source_tokens[:]
        max_length = self.max_source_tokens_count + self.max_target_tokens_count + 2

        target_tokens = self.tokenizer(
            target,
            add_special_tokens=False,
            max_length=self.max_target_tokens_count,
            padding=False,
            truncation=True,
        )["input_ids"]

        input_ids += target_tokens + [self.tokenizer.eos_token_id]

        if self.padding:
            actual_length = len(input_ids)
            padding = [
                self.tokenizer.pad_token_id for i in range(len(input_ids), max_length)
            ]
            input_ids.extend(padding)

        input_ids = torch.LongTensor(input_ids)
        labels = input_ids.clone()
        attention_mask = input_ids.new_ones(input_ids.size())

        if self.padding:
            labels[actual_length:] = -100
            attention_mask[actual_length:] = 0

        if self.only_target_loss:
            labels[: len(source_tokens)] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class BaseAdapter:
    """The base and the default model adapter."""

    def match(self, model_path: str):
        return True

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("one_shot")


class IEasQAAdapter(BaseAdapter):
    """The model adapter for FreedomIntelligence/phoenix-inst-chat-7b"""

    def match(self, model_path: str):
        return "ie_as_qa" in model_path

    def load_model(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("ie_as_qa")


model_adapters: list[BaseAdapter] = []


@cache
def get_model_adapter(model_path: str) -> BaseAdapter:
    """Get a model adapter for a model_path."""
    for adapter in model_adapters:
        if adapter.match(model_path):
            return adapter
    raise ValueError(f"No valid model adapter for {model_path}")


def get_conversation_template(model_path: str) -> Conversation:
    adapter = get_model_adapter(model_path)
    return adapter.get_default_conv_template(model_path)


def preprocess2(
    sources,
    tokenizer: PreTrainedTokenizer,
) -> dict:
    conv = get_conversation_template("ie_as_qa")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    input_ids, labels = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv_input_ids, conv_targets = [], []

        # Randomly shuffle the queries, since the answer should be order-insensitive to queries.
        assert len(source) % 2 == 0
        idxs = list(range(len(source) // 2))[1:]
        random.shuffle(idxs)
        idxs = [0] + idxs
        new_source = []
        for rand_i in idxs:
            new_source.append(source[2 * rand_i])
            new_source.append(source[2 * rand_i + 1])
        source = new_source

        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if j == 0:
                # first input consists of system message and user input
                message = conv.system + conv.sep + role + ": " + sentence["value"]
                _input_ids = tokenize(tokenizer, message, True)
                conv_input_ids += _input_ids
                conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
            else:
                if j % 2 == 0:
                    # user input
                    message = role + ": " + sentence["value"]
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
                else:
                    # assistant output
                    message = role + ": "
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += [IGNORE_TOKEN_ID] * len(_input_ids)
                    message = sentence["value"] + conv.sep2
                    _input_ids = tokenize(tokenizer, message)
                    conv_input_ids += _input_ids
                    conv_targets += _input_ids

        assert len(conv_input_ids) == len(conv_targets), source

        if len(conv_input_ids) < tokenizer.model_max_length:
            pad_length = tokenizer.model_max_length - len(conv_input_ids)
            conv_input_ids += [tokenizer.pad_token_id] * pad_length
            conv_targets += [IGNORE_TOKEN_ID] * pad_length

        input_ids.append(conv_input_ids[: tokenizer.model_max_length])
        labels.append(conv_targets[: tokenizer.model_max_length])

    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
