# Evaluating Factuality Word Definitions

## Table of Contents
- [Setup](#setup)
  - [Python Environment Setup](#1-python-environment-setup)
  - [Install Required Packages](#2-install-required-packages)
  - [Configure Project Settings](#3-configure-project-settings)
  - [Discourse Simplification Repository Setup](#4-discourse-simplification-repository-setup)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Data](#data)
  - [Download Evaluation & Dataset Data](#download-evaluation--dataset-data)
  - [Download Wiki Pages (Optional)](#download-wiki-pages-optional)
- [Authors and Acknowledgments](#authors-and-acknowledgments)

## Setup

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
For the discourse simplification component, clone the required repository and check out the specific commit as follows:
```
cd ..
git clone git@github.com:Lambda-3/DiscourseSimplification.git
cd DiscourseSimplification
git checkout 5e7ac12
mvn clean install -DskipTests
```

## Usage
TODO show pipeline object
and notebooks

## Repository Structure
add explainations where to fidn everything

## Data
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

## Authors and Acknowledgments
Lukas Ellinger (lukas.ellinger@tum.de)