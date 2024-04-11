"""Test file."""

from config import PROJECT_DIR, DB_URL
from datasets import load_dataset, Dataset


train_file_path = PROJECT_DIR.joinpath("dataset/def_train.jsonl")
dev_file_path = PROJECT_DIR.joinpath("dataset/def_dev.jsonl")
test_file_path = PROJECT_DIR.joinpath("dataset/def_test.jsonl")

train_dataset = load_dataset("json", data_files=str(train_file_path))["train"]
#dev_dataset = load_dataset("json", data_files=str(dev_file_path))["train"]
#test_dataset = load_dataset("json", data_files=str(test_file_path))["train"]

dataset = Dataset.from_sql("SELECT * FROM def_dataset WHERE set_type='train'", con=DB_URL)
