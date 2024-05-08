import os
from setuptools import setup, find_packages  # or find_namespace_packages

CWD = os.getcwd()
print("Current working directory: ", CWD)

os.system(f'echo NERLLAMA_SOURCE={CWD} >> ~/.bash_profile; echo export NERLLAMA_SOURCE')

if os.path.exists('README.md'):
    long_description = open('README.md').read()
else:
    long_description = "Chemical Named Entity Recognition Training and Testing Tool"

setup(name='ChemInstruct',
    version='1.1.0',
    author='Madhavi Kumari',
    author_email='xyzmadhavi@gmail.com',
    license='MIT',
    url='https://github.com/chemplusx/ChemInstruct',
    packages=find_packages(),
    description='A toolkit for Training and Testing LLMs on CNER tasks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='text-mining mining chemistry cheminformatics nlp html xml science scientific',
    zip_safe=False,
    entry_points={'console_scripts': ['nerl = nerllama:cli']},
    tests_require=['pytest'],
    install_requires=[
        'datasets==2.16.1',
        'huggingface_hub==0.20.2',
        'langchain==0.1.5',
        'llama_index==0.7.21',
        'numpy==1.26.3',
        'pandas==2.2.0',
        'peft==0.7.1',
        'scikit_learn==1.4.0',
        'streamlit==1.30.0',
        'torch==2.1.2',
        'tqdm==4.66.1',
        'transformers==4.37.0',
        # 'vllm==0.2.7',
        'wandb==0.16.2'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Text Processing :: Markup :: HTML',
    ],
)
