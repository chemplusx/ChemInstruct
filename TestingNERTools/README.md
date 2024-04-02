# Introduction
TestingNERTools is a project to test the available NER tools on the market. The project is designed to be easy to use and flexible, allowing users to easily test the supported tools in the project.

# Installation
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

# Build / Run

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



