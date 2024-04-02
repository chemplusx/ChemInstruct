from argparse import ArgumentParser

from NERLLaMA.src.common.constants import LLAMA_MODEL_MAP


def run_model(model, text, inputFile, pipeline, max_new_tokens, auth_token):
    if not pipeline or pipeline != "RAG":
        if model in ["llama2-base-ft", "base-ft"]:
            from NERLLaMA.src.fine_tuned.llama2_base_ft import run

            full_model_name = LLAMA_MODEL_MAP["llama2-base-ft"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["llama2-chat-ft", "chat-ft"]:
            from NERLLaMA.src.fine_tuned.llama2_chat import run

            full_model_name = LLAMA_MODEL_MAP["llama2-chat-ft"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["llama2-chat", "chat"]:
            from NERLLaMA.src.fine_tuned.llama2_chat import run

            full_model_name = LLAMA_MODEL_MAP["llama2-chat"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["llama2-chat-70b", "chat"]:
            from NERLLaMA.src.fine_tuned.llama2_chat import run

            full_model_name = LLAMA_MODEL_MAP["llama2-chat-70b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["mistral-chat-7b", "chat"]:
            from NERLLaMA.src.fine_tuned.mistral import run

            full_model_name = LLAMA_MODEL_MAP["mistral-chat-7b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["falcon-chat-7b", "chat"]:
            from NERLLaMA.src.fine_tuned.falcon import run

            full_model_name = LLAMA_MODEL_MAP["falcon-chat-7b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
        elif model in ["falcon-chat-180b", "chat"]:
            from NERLLaMA.src.fine_tuned.falcon import run

            full_model_name = LLAMA_MODEL_MAP["falcon-chat-180b"]
            return run(full_model_name, text, inputFile, max_new_tokens, auth_token)
    else:
        full_model_name = None
        if model in ["llama2-base-ft", "base-ft"]:
            full_model_name = "ChemPlusX/llama2-base-ft-NER"
        elif model in ["llama2-chat-ft", "chat-ft"]:
            full_model_name = "ChemPlusX/llama2-chat-ft-NER"
        elif model in ["llama2-chat", "chat"]:
            full_model_name = "meta-llama/Llama-2-7b-chat-hf"

        from NERLLaMA.src.rag.generate import run

        return run(full_model_name, text, inputFile, max_new_tokens, auth_token)


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
    parser.add_argument("--max_tokens", default=256, type=int, help="Max output tokens")
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
