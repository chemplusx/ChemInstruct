# Introduction
This project ChemInstruct has 3 main components:
1. TestinNERTools: Testing NER Tools on the NLMChem and ChemInstruct dataset
2. NERLLaMA: Training and evaluating LLaMA models on the NLMChem and ChemInstruct dataset
3. Dataset: All the datasets used in the entire project

# Installation
Installation for both the components is different and the same is mentioned in the respective folders.

## NERLLaMA

### Introduction
NERLLaMA is a Named Entity Recognition (NER) tool that uses a machine learning approach to identify named entities in text. The tool uses the power of Large Language Models (LLMs). The tool is designed to be easy to use and flexible, allowing users to train and evaluate models on their own data.

### Installation
To install NERLLaMA, you will need to have Python 3.9 or higher installed on your system.
To install NERLLaMA, navigate to ChemInstruct/NERLLaMA, and run the following command:

```bash
pip install -e .
```

### Usage

The tool / package can be used as below

1: As package:

This installs nerllama package in your active python venv, or conda env.
Hence the package can be directly used in your own custom code.

```python
from nerllama.schemas.ChemStruct import InstructDataset

id = InstructDataset()
id.convert_instruction_causal()
```

2: From CLI

Once the above installation completes. a `nerl` CLI interface is also available to be accessed from the terminal.
This cli command facilitates quick and easy use of the nerllama commands to extract entities etc.
Check [CLI-Interaction](#cli-interaction), for more details on how to use the command.


NERLLaMA uses the Hugging Face Transformers library to work with LLMs. You will need to have an account on the Hugging Face website to use the tool. You can sign up for an account [here](https://huggingface.co/join).
We have fine-tuned and evaluated the pre=trained models over GPU. Hence the project requires [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) to be installed on your system.

LLaMA models can be accessed only after getting access to the models from Meta AI portal and Hugging Face.
The same can be requested from the [LLaMA HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b).

### Usage
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


We have used [W&B](https://wandb.ai/) to collect and sync generation / training data.
When using CLI, you might be prompted for connection to W&B.

```bash
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 3
```

When asked for choice, enter 3, to skip connecting to W&B


### CLI-Interaction

NERLLaMA exposes a `nerl` cli command for easy access of the tolls functionalities

Running `nerl nerllama` command to extract Chemical Entities from given file

```
nerl nerllama run "<path to file containing chemical literature>" <model HF path / or shorthand (mentioned above)> <pipeline: LLM/RAG> <hf token>
```

#### Sample:

* Predefined Models:
```
nerl nerllama run /home/ubuntu/data/sample_text.txt llama2-chat-ft LLM hf_*****
```

* Any new Model (Chat based)
```
nerl nerllama run /home/ubuntu/data/sample_text.txt meta-llama/Meta-Llama-3-8B-Instruct LLM hf_*****
```

* Running with RAG:
```
nerl nerllama run /home/ubuntu/data/sample_text.txt llama2-chat-ft RAG hf_*****
```


## TestingNERTools

### Introduction
TestingNERTools is a project to test the available NER tools on the market. The project is designed to be easy to use and flexible, allowing users to easily test the supported tools in the project.

### Installation
The project is divided into two parts:
The first part is java based and the second part is python based.

**First:**
To install the java part, you will need to have Java 8 or higher installed on your system.

Download the following files:

* **Chemspot**: https://drive.google.com/file/d/1oYmpO_HaD7ygCu1o4l8wm6iQo6NpiEvL/view?usp=drive_link
* **CheNER**: https://drive.google.com/file/d/1tqFxwYTOwXfCxJt4IJ7EKMuZKn_yQyIJ/view?usp=drive_link
* **javaFX**: https://drive.google.com/file/d/1Nj8Eomgr0c3o5S5fSPX9bvC0Gl71fZ5F/view?usp=drive_link


Move all the above downloaded files into the packages folder.

Extract javafx-sdk-21.zip in packages folder

### Build / Run

To build the project, run the following command:
```bash
javac -cp ".;<root directory>\ChemInstruct\TestingNERTools\packages\*;<root directory>\ChemInstruct\TestingNERTools\src\"  <root directory>\ChemInstruct\TestingNERTools\src\StartEvaluation.java
```

```bash
java -cp ".;<root directory>\ChemInstruct\TestingNERTools\packages\*;<root directory>\ChemInstruct\TestingNERTools\src\"  <root directory>\ChemInstruct\TestingNERTools\src\StartEvaluation.java  --directory <input directory path> --tool <tool name> --dataset <dataset>
```

**Arguments:**
* dataset: nlmchem / custom
* tool: chener / chemspot



**Second:**
To install the python part, you will need to have Python 3.9 or higher installed on your system.

To install all the dependencies, run the following command:
```bash
cd python_src
pip install -r requirements.txt
```


# Usage
Usage for both the components is different and the same is mentioned in the respective folders.

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

# Acknowledgements
We would like to thank the Hugging Face team for providing the infrastructure and tools that made this project possible. We would also like to thank the community for their support and contributions.

# References
- [Hugging Face](https://huggingface.co/)
- [Transformers](https://huggingface.co/docs/transformers/en/index.html)
- [LLaMA2](https://huggingface.co/meta-llama/Llama-2-7b)
- [NLMChem]()

