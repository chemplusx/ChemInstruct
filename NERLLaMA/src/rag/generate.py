# Import streamlit for app dev
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import pandas as pd
from pathlib import Path

from NERLLaMA.src.common.constants import (
    LLAMA_MODELS,
    AUTH_TOKEN_REQUIREMENT_ERROR,
)
from NERLLaMA.src.schemas.DataStruct import INSTRUCTION_TEXT_RAG, SYSTEM_PROMPT


@st.cache_resource
def get_tokenizer_model(name, auth_token):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_auth_token=auth_token)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        name,
        use_auth_token=auth_token,
        torch_dtype=torch.float16,
        rope_scaling={"type": "dynamic", "factor": 2},
        load_in_8bit=True,
    )

    return tokenizer, model


def generate_with_rag(text):
    inst = f"{INSTRUCTION_TEXT_RAG}"
    with open(f"./data/file.txt", "w") as f:
        f.write(text)
    # Create an index - we'll be able to query this in a sec
    print("Loading in vector store index...")
    dd = SimpleDirectoryReader(input_dir="./data/").load_data()
    index = VectorStoreIndex.from_documents(dd)
    # Setup index query engine using LLM
    print("setting up query engine...")
    query_engine = index.as_query_engine()

    response = query_engine.query(inst)
    print("response: ", response)
    # ...and write it out to the screen
    st.write(response)

    # Display raw response object
    with st.expander("Response Object"):
        st.write(response)
    # Display source text
    with st.expander("Source Text"):
        st.write(response.get_formatted_sources())


def run(
    model=None,
    text=None,
    dataset_file=None,
    max_new_tokens=256,
    auth_token=None,
    context_window=4096,
    **kwargs,
):
    if not model or model not in LLAMA_MODELS:
        raise Exception(f"Invalid model. Model: {model}")
    if not auth_token:
        raise Exception(f"Invalid/Empty Auth Token. {AUTH_TOKEN_REQUIREMENT_ERROR}")
    if not dataset_file or not Path(dataset_file).exists() and not text:
        raise Exception(f"One of Text or dataset_file is required")
    else:
        tokenizer, model = get_tokenizer_model(model, auth_token)

        # Create a system prompt
        system_prompt = f"<s>[INST] <<SYS>>{SYSTEM_PROMPT}<</SYS>>"
        # Throw together the query wrapper
        query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

        # Create a HF LLM using the llama index wrapper
        llm = HuggingFaceLLM(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            query_wrapper_prompt=query_wrapper_prompt,
            model=model,
            tokenizer=tokenizer,
        )

        # Create and dl embeddings instance
        embeddings = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        # Create new service context instance
        service_context = ServiceContext.from_defaults(
            chunk_size=1024, llm=llm, embed_model=embeddings
        )
        # And set the service context
        set_global_service_context(service_context)

        if text:
            generate_with_rag(text)
        else:
            data = pd.read_csv(dataset_file, encoding="cp1252")
            inputs = data.get("text")
            for i in range(len(inputs)):
                generate_with_rag(inputs[i])
