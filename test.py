"""Test file."""

from config import PROJECT_DIR
from datasets import load_dataset

dataset = load_dataset('json', data_files='datasets/def_dev.jsonl')

train_file_path = PROJECT_DIR.joinpath("datasets/def_train.jsonl")
dev_file_path = PROJECT_DIR.joinpath("datasets/def_dev.jsonl")
test_file_path = PROJECT_DIR.joinpath("datasets/def_test.jsonl")

train_dataset = load_dataset("json", data_files=str(train_file_path))["train"]
dev_dataset = load_dataset("json", data_files=str(dev_file_path))["train"]
test_dataset = load_dataset("json", data_files=str(test_file_path))["train"]
