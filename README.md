# Evaluating the Factuality of Word Definitions with a Multilingual RAE-Pipeline
![Python Version](https://img.shields.io/badge/python-3.10-blue)<br>
This repository contains the code and resources for evaluating the factual accuracy of word 
definitions using our multilingual Retrieval-Augmented Evaluation Pipeline (RAE-Pipeline). 

With the growing reliance on Large Language Models (LLMs) like OpenAI's GPT series, ensuring 
the accuracy of generated content—especially word definitions—has become increasingly important. 
Definitions are foundational to effective communication and learning, and inaccuracies can lead 
to misunderstandings and misinformation. Our focus is on non-English languages, particularly German, 
which are often underrepresented in fact-checking advancements. 

## 📑 Table of Contents
- [⚙️ Setup](#-setup)
  - [Python Environment Setup](#1-python-environment-setup)
  - [Install Required Packages](#2-install-required-packages)
  - [Configure Project Settings](#3-configure-project-settings)
  - [Discourse Simplification Repository Setup](#4-discourse-simplification-repository-setup)
- [🚀 Usage](#-usage)
- [📂 Repository Structure](#-repository-structure)
- [📊 Data](#-data)
  - [Download Evaluation & Dataset Data](#download-evaluation--dataset-data)
  - [Download Wiki Pages (Optional)](#download-wiki-pages-optional)
- [🤝 Authors and Acknowledgments](#-authors-and-acknowledgments)

## ⚙️ Setup
To set up the project environment, follow these steps:

### 1. Python Environment Setup
Make sure you have Python 3.10 or above installed. 
It's recommended to use a virtual environment to avoid conflicts with other projects. 
You can create and activate a virtual environment by running:
```
python -m venv evaluating-word-definitions
source evaluating-word-definitions/bin/activate  # On Windows use: evaluating-word-definitions\Scripts\activate
```

### 2. Install Required Packages
Once the virtual environment is activated, install the necessary dependencies:
```
pip install -r requirements.txt
python setup.py
```
### 3. Configure Project Settings
To properly configure the project, set up the configuration file:
```
cp config.py.template config.py
nano config.py  # Use any text editor to modify config.py
```
Ensure to add all necessary attributes in config.py, such as API keys, model paths, or dataset locations.

### 4. Discourse Simplification Repository Setup
This step is optional and required only if you plan to use the Discourse Simplification component (DisSim Splitter). 
Otherwise, precomputed fact splits are already included in the datasets. To set it up:
```
cd ..
git clone git@github.com:Lambda-3/DiscourseSimplification.git
cd DiscourseSimplification
git checkout 5e7ac12
mvn clean install -DskipTests
```

## 🚀 Usage

Below is an example of how to use the RAE-Pipeline to verify the factual accuracy of word definitions. The pipeline consists of various modules, including translation, sentence connection, evidence fetching, and verification. You can adjust the components based on your use case.

### Example 1: Verifying a Single Claim
In this example, we verify a claim about the word "unicorn" using the pipeline.

```python
from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.pipeline import Pipeline
from pipeline_module.sentence_connector import ColonSentenceConnector
from pipeline_module.statement_verifier import ModelStatementVerifier
from pipeline_module.translator import OpusMTTranslator

# Initialize the pipeline
pipeline = Pipeline(
    translator=OpusMTTranslator(),
    sent_connector=ColonSentenceConnector(), # or PhiSentenceConnector()
    claim_splitter=None, # or DisSimSplitter() | T5SplitRephraseSplitter() | FactscoreSplitter()
    evid_fetcher=WikipediaEvidenceFetcher(offline=False),
    evid_selector=ModelEvidenceSelector(),
    stm_verifier=ModelStatementVerifier(),
    lang='en'
)

# Verifying a claim
result = pipeline.verify(
    word='unicorn',
    claim='mythical horse with a single horn'
)
print(result)  # Displays the verification result
```

### Example 2: Verifying a Dataset of Claims

This example demonstrates how to verify multiple claims from a dataset. We load the dataset from Hugging Face and use the pipeline to verify all claims.

```python
from datasets import load_dataset

from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.pipeline import Pipeline
from pipeline_module.sentence_connector import ColonSentenceConnector
from pipeline_module.statement_verifier import ModelStatementVerifier
from pipeline_module.translator import OpusMTTranslator

# Initialize the pipeline
pipeline = Pipeline(
    translator=OpusMTTranslator(),
    sent_connector=ColonSentenceConnector(), # or PhiSentenceConnector()
    claim_splitter=None, # or DisSimSplitter() | T5SplitRephraseSplitter() | FactscoreSplitter()
    evid_fetcher=WikipediaEvidenceFetcher(offline=True),
    evid_selector=ModelEvidenceSelector(),
    stm_verifier=ModelStatementVerifier(),
    lang='en'
)
dataset = load_dataset('lukasellinger/german_dpr-claim_verification', split='test')

outputs, report, not_in_wiki = pipeline.verify_test_dataset(dataset)
print(report) # Displays the classification report
```

### Key Modules and Options:

- **Translator**: Handles translation of input claims for multilingual support. In this example, we use `OpusMTTranslator()`, which by default translates from German (`source_lang='de'`) to English (`dest_lang='en'`). You can adjust these parameters to suit your specific language needs.
- **Sentence Connector**: Modules like `ColonSentenceConnector()` and `PhiSentenceConnector()` are used to combine claims or evidence for better context.
- **Evidence Fetcher**: Retrieves evidence from Wikipedia. It can run in `offline` mode (using a local dump) or `online` mode (fetching live data from Wikipedia API).
- **Claim Splitter**: Optional module for splitting complex claims into simpler ones.  You can use `None`, `DisSimSplitter()`, `T5SplitRephraseSplitter()` and `FactscoreSplitter()`.
- **Evidence Selector**: This module ranks and selects the most relevant pieces of evidence for verifying the claim.
- **Statement Verifier**: Verifies the factual accuracy of the claim using the selected evidence.

## 📂 Repository Structure
```
📁 evaluating-word-definitions/
├── 📁 database/                          # Related to Sqlite DB (not needed)
├── 📁 dataset/                           # Datasets used for training
├── 📁 factscore/                         # Adapted FActScore repository
├── 📁 fetchers/                          # Modules for retrieving data from Wikipedia and OpenAI APIs
├── 📁 general_utils/                     # Utility functions and helpers
├── 📁 graphics/                          # Visualizations and graphical outputs used in paper
├── 📁 losses/                            # Custom loss functions for training
├── 📁 models/                            # Custom models
├── 📁 notebooks/                         # Jupyter notebooks for experiments and analysis
│   ├── create_datasets.ipynb             # Example usage of the pipeline
│   ├── evaluation_factscore.ipynb        # Experiments with FActScore
│   ├── evaluation_openai.ipynb           # Experiments with OpenAi models
│   ├── evaluation_pipeline.ipynb         # Experiments of our Pipeline
│   ├── fintune_claim_verification.ipynb  # Fine-tuning script of claim verification model
│   ├── finetune_evidence_selection.ipynb # Fine-tuning script of evidence selection model
│   └── stats.ipynb                       # Gather stats of the datasets
├── 📁 pipeline_module/                   # Including the pipeline and its modules
├── 📁 scripts/                           # Various scripts
├── 📁 training_loop_tests/               # Tests of training scripts
├── config.py.template                    # Template of config.py
├── README.md                             # Project documentation (this file)
├── requirements.txt                      # List of dependencies
└── setup.py                              # Package setup script
```

## 📊 Data
The evaluation data is too large to be stored in this Git repository. 
However, downloading it is not required to run the pipeline. 
If needed, you can manually download the data from the following links:
[Link to Data](https://drive.google.com/drive/folders/1Vj15MWmNMzld7odGNZ9h7B9qrn4OKehV?usp=drive_link)

### Download Evaluation & Dataset Data
To download the evaluation dataset, use gdown to fetch the data from Google Drive:
```bash
gdown --folder "https://drive.google.com/drive/folders/1E8Vi6zmTDldCdWHrWReVx5teMtt0cite?usp=drive_link" -O data
```
This will download all the necessary files to the data/ folder.

### Download Wiki Pages (Optional)
We also provide the Wiki Pages used in the FEVER task. 
If you need them for your analysis, you can download the Wiki Pages using the following command:
```bash
gdown --folder "https://drive.google.com/drive/folders/1FUfz6101wAFPWUEyEMPhyHJegA3uHcZM?usp=drive_link" -O wiki-pages
```

## 🤝 Authors and Acknowledgments
Lukas Ellinger (lukas.ellinger@tum.de)