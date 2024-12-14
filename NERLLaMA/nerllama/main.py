from argparse import ArgumentParser
import click
from pathlib import Path

from nerllama.common.constants import LLAMA_MODEL_MAP, AUTH_TOKEN_REQUIREMENT_ERROR


def run_model(model, text, inputFile, pipeline, max_new_tokens, auth_token):
    if not pipeline or pipeline != "RAG":
        if model in ["llama2-base-ft", "base-ft"]:
            from nerllama.fine_tuned.llama2_base_ft import run

            full_model_name = LLAMA_MODEL_MAP["llama2-base-ft"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["llama2-chat-ft", "chat-ft"]:
            from nerllama.fine_tuned.llama2_chat import run

            full_model_name = LLAMA_MODEL_MAP["llama2-chat-ft"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["llama2-chat", "chat"]:
            from nerllama.fine_tuned.llama2_chat import run

            full_model_name = LLAMA_MODEL_MAP["llama2-chat"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["llama2-chat-70b", "chat"]:
            from nerllama.fine_tuned.llama2_chat import run

            full_model_name = LLAMA_MODEL_MAP["llama2-chat-70b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["mistral-chat-7b", "chat"]:
            from nerllama.fine_tuned.mistral import run

            full_model_name = LLAMA_MODEL_MAP["mistral-chat-7b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["falcon-chat-7b", "chat"]:
            from nerllama.fine_tuned.falcon import run

            full_model_name = LLAMA_MODEL_MAP["falcon-chat-7b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["falcon-chat-180b", "chat"]:
            from nerllama.fine_tuned.falcon import run

            full_model_name = LLAMA_MODEL_MAP["falcon-chat-180b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        else:
            from nerllama.fine_tuned.misc import run
            return run(model, text, inputFile, max_new_tokens, auth_token)
    else:
        full_model_name = None
        if model in ["llama2-base-ft", "base-ft"]:
            full_model_name = "ChemPlusX/llama2-base-ft-NER"
        elif model in ["llama2-chat-ft", "chat-ft"]:
            full_model_name = "ChemPlusX/llama2-chat-ft-NER"
        elif model in ["llama2-chat", "chat"]:
            full_model_name = "meta-llama/Llama-2-7b-chat-hf"

        from nerllama.rag.generate import run

        return run(full_model_name, text, inputFile, max_new_tokens, auth_token)

"""
@click.group(name='chemdner')
@click.pass_context
def chemdner_cli(ctx):
    CHEMDNER commands.
    pass


@chemdner_cli.command()
@click.argument('annotations', type=click.File('r', encoding='utf8'), required=True)
@click.option('--gout', '-g', type=click.File('w', encoding='utf8'), 
help='Gold annotations output.', required=True)
@click.pass_obj
def prepare_gold(ctx, annotations, gout):
    #Prepare bc-evaluate gold file from annotations supplied by CHEMDNER.
    click.echo('chemdataextractor.chemdner.prepare_gold')
    for line in annotations:
        pmid, ta, start, end, text, category = line.strip().split('\t')
        gout.write('%s\t%s:%s:%s\n' % (pmid, ta, start, end))
"""


@click.group(name="nerllama")
@click.pass_context
def nerllama_cli(ctx):
    """ NERLLaMA Cli Commands """
    pass

@nerllama_cli.command()
@click.argument('file', type=click.File(mode = "r", encoding="utf-8"), required=False)
@click.argument('model', type=click.STRING, required=True)
@click.argument('pipeline', type=click.STRING, default="LLM")
@click.argument('auth_token', type=click.STRING)
@click.pass_obj
def run(ctx, file, model, pipeline, auth_token):
    click.echo('nerllama.nerllama.run')
    if not model or model == "":
        raise Exception(f"Invalid model. Model: {model}")
    if not auth_token:
        raise Exception(f"Invalid/Empty Auth Token. {AUTH_TOKEN_REQUIREMENT_ERROR}")
    if (not file):
        raise Exception(f"Invalid/Empty Dataset file. File: {file}")
    
    run_model(
        model,
        "",
        file,
        pipeline,
        1024,
        auth_token,
    )
    return



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--text",
        default="I need real calcium.",
        type=str,
        
        help="Text to be processed. Not required if file is provided.",
    )
    parser.add_argument(
        "--file",
        default=None,
        type=str,
        
        help="Name of the file. Not required if text is provided.",
    )
    parser.add_argument(
        "--model",
        default="base-ft",
        type=str,
        
        help="Name of the model. Options: [base-ft, chat-ft, chat]",
    )
    parser.add_argument(
        "--pipeline",
        default="LLM",
        type=str,
        
        help="RAG or Simple LLM. Options: [RAG, LLM]",
    )
    parser.add_argument("--max_tokens", default=256, type=int, 
    help="Max output tokens")
    parser.add_argument(
        "--auth_token",
        default="<hf_token>",
        type=str,
        
        help="HuggingFace Auth Token. Required for LLaMA models.",
    )

    args = parser.parse_args()

    run_model(
        args.model,
        args.text,
        args.file,
        args.pipeline,
        args.max_tokens,
        args.auth_token,
    )
