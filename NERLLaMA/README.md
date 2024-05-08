# Introduction
NERLLaMA is a Named Entity Recognition (NER) tool that uses a machine learning approach to identify named entities in text. The tool uses the power of Large Language Models (LLMs). The tool is designed to be easy to use and flexible, allowing users to train and evaluate models on their own data.

# Installation
To install NERLLaMA, you will need to have Python 3.9 or higher installed on your system.

NERLLaMA gets installed as a package also exposes command line interface to inteact with its components

To install NERLLaMA, navigate to ChemInstruct/NERLLaMA, and run the following command:

```bash
pip install -e .
```

Once the above installation completes. a `nerl` CLI interface is available to be accessed from the terminal.

**Check CLI-Interaction, for more details on how to use the command.**

NERLLaMA uses the Hugging Face Transformers library to work with LLMs. You will need to have an account on the Hugging Face website to use the tool. You can sign up for an account [here](https://huggingface.co/join).
We have fine-tuned and evaluated the pre=trained models over GPU. Hence the project requires [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) to be installed on your system.

LLaMA models can be accessed only after getting access to the models from Meta AI portal and Hugging Face.
The same can be requested from the [LLaMA HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b).

# Usage
To use NERLLaMA, you will need to have a trained model. In this project we provide a pre-trained model that you can use to get started. To use the pre-trained model, run the following command:

```bash
python main.py --text "Your text goes here" --model "llama2-chat-ft" --pipeline "llm" --auth_token "<your huggingface auth token>"
```

```bash
python main.py --file "path/to/inputfile" --model "llama2-chat-ft" --pipeline "llm" --auth_token "<your huggingface auth token>"
```

models:
- `llama2-chat-ft` - LLaMA2 Chat Fine-Tuned
- `llama2-base-ft` - LLaMA2 base Fine-Tuned
- `llama2-chat` - LLaMA2 Chat HF
- `llama2-chat-70b` - LLaMA2 Chat HF 70B
- `mistral-chat-7b` - MistralAI 7B Instruct v0.2
- `falcon-chat-7b` - TII's Falcon 7b Instruct

pipelines:
- `llm` - Large Language Model
- `rag` - Retrieval Augmented Generation

# Contributing
If you would like to contribute to NERLLaMA, please open an issue or submit a pull request. We welcome contributions from the community.

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

# Acknowledgements
We would like to thank the Hugging Face team for providing the infrastructure and tools that made this project possible. We would also like to thank the community for their support and contributions.

# References
- [Hugging Face](https://huggingface.co/)
- [Transformers](https://huggingface.co/docs/transformers/en/index.html)
- [LLaMA2](https://huggingface.co/meta-llama/Llama-2-7b)

# Contact
If you have any questions or comments, please feel free to reach out to us at [
