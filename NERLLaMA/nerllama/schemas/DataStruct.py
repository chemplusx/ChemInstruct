import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from typing import Union

from nerllama.schemas.ChemStruct import ChemStruct
from nerllama.schemas.Misc import Record
from transformers import PreTrainedTokenizer
from nerllama.utils.train_utils import (
    create_output_from_entities,
    MODEL_INPUT_TEMPLATE,
)
from nerllama.utils.utils import load_lines, parse_jsonl

from nerllama.schemas.ChemStruct import preprocess2
from nerllama.schemas.Conversation import rank0_print

ENTITY_TYPES = ["chemical"]
ENTITY_DEFENITIONS = [
    "mention of drugs, organic compounds, protiens and any other chemical substances"
]
INSTRUCTION_TEXT = "You are solving the NER problem. Extract from the text words related to each of the following entities: chemical"
INSTRUCTION_TEXT_RAG = """
From the text, extract all the exact mentions of chemical entities. (A chemical entity can be drugs, organic compounds, protiens, enzymes and any other chemical substances)
"""

SYSTEM_PROMPT = """
You are solving the Named Entity Recognition problem.
Only answer the question based on the {context} above. 
If the {question} is not in the {context}, just say that you don’t know the answer, don’t try to make up an answer.
Respond with only the entity text.
"""


class AnnotatedDataset(Record):
    __attributes__ = ["file_name", "text", "entities"]

    def __init__(self, file_name, text, entities):
        self.file_name = file_name
        self.text = text
        self.entities = entities


class AnnotatedEntity(Record):
    __attributes__ = ["entity_id", "entity_text", "entity_type", "start", "end"]

    def __init__(self, entity_id, entity_text, entity_type, start, end):
        self.entity_id = entity_id
        self.entity_text = entity_text
        self.entity_type = entity_type
        self.start = start
        self.end = end


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess2(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess2([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def parse_entities(items):
    for item in items:
        yield AnnotatedEntity(
            item["entity_id"],
            item["entity_text"],
            item["entity_type"],
            item["start"],
            item["end"],
        )


def parse_annotated_data(items):
    for item in items:
        entities = list(parse_entities(item["entities"]))
        yield AnnotatedDataset(item["file_name"], item["text"], entities)


def load_annotated_data(path):
    lines = load_lines(path)
    items = parse_jsonl(lines)
    return parse_annotated_data(items)


def entity_type_to_instruction(entity_type: str) -> str:
    base_phrase = "You are solving the NER problem. Extract from the text "
    return base_phrase + dict(zip(ENTITY_TYPES, ENTITY_DEFENITIONS))[entity_type]


def parse_entities_from_record(record: AnnotatedDataset) -> tuple[str, dict[str, list]]:
    entities = dict(zip(ENTITY_TYPES, [[] for _ in range(len(ENTITY_TYPES))]))
    for entity in record.entities:
        entities[entity.entity_type].append(entity.entity_text)

    return record.text, entities


def create_instructions_for_record(
    record: AnnotatedDataset, is_separate_labels: bool = False
) -> Union[list[ChemStruct], ChemStruct]:
    text, entities = parse_entities_from_record(record)
    if is_separate_labels:
        record_instructions = []
        for entity_type in entities.keys():
            instruction = entity_type_to_instruction(entity_type)
            record_instructions.append(
                {
                    "instruction": instruction,
                    "input": text,
                    "output": create_output_from_entities(entities[entity_type]),
                    "source": MODEL_INPUT_TEMPLATE["prompts_input"].format(
                        instruction=instruction.strip(), inp=text.strip()
                    ),
                    "label": entity_type,
                    "id": f"{record.sentence_id}_{record.file_name}",
                }
            )
        return record_instructions
    else:
        return {
            "instruction": INSTRUCTION_TEXT,
            "input": text,
            "output": create_output_from_entities(entities, out_type=2),
            "source": MODEL_INPUT_TEMPLATE["prompts_input"].format(
                instruction=INSTRUCTION_TEXT.strip(), inp=text.strip()
            ),
            "raw_entities": entities,
            "id": f"{record.file_name}",
        }


def _fill_instructions_list(
    dataset: list[AnnotatedDataset], is_separate_labels: bool = False
) -> list[ChemStruct]:
    instructions = []
    for record in tqdm(dataset):
        if is_separate_labels:
            instructions = np.concatenate(
                (
                    instructions,
                    create_instructions_for_record(record, is_separate_labels),
                )
            )
        else:
            instructions.append(
                create_instructions_for_record(record, is_separate_labels)
            )

    return instructions


def create_instruct_dataset(
    data_path: str, max_instances: int = -1, is_separate_labels: bool = False
) -> list[ChemStruct]:
    rudrec_dataset = list(load_annotated_data(data_path))

    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    return _fill_instructions_list(rudrec_dataset, is_separate_labels)


def create_train_test_instruct_datasets(
    data_path: str,
    max_instances: int = -1,
    is_separate_labels: bool = False,
    test_size: float = 0.3,
    random_seed: int = 42,
) -> tuple[list[ChemStruct], list[ChemStruct]]:
    rudrec_dataset = list(load_annotated_data(data_path))

    if max_instances != -1 and len(rudrec_dataset) > max_instances:
        rudrec_dataset = rudrec_dataset[:max_instances]

    train_dataset, test_dataset = train_test_split(
        rudrec_dataset, test_size=test_size, random_state=random_seed
    )
    return _fill_instructions_list(
        train_dataset, is_separate_labels
    ), _fill_instructions_list(test_dataset, is_separate_labels)
