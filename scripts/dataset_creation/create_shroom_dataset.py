"""Script for the creation of the SHROOM dataset."""
import re

from datasets import DatasetDict, load_dataset

from config import HF_WRITE_TOKEN, PROJECT_DIR

data_files = {
    'modelagnostic': str(PROJECT_DIR / "dataset/SHROOM_test-labeled/test.model-agnostic.json"),
    'modelaware': str(PROJECT_DIR / "dataset/SHROOM_test-labeled/test.model-aware.json")
}

dataset = load_dataset('json', data_files=data_files)
dataset = dataset.filter(lambda e: e['task'] == 'DM')

a = dataset['modelagnostic']
b = dataset['modelaware']


def extract_word(example):
    """Extract the word of the src sentence."""
    # Use regex to find the word inside <define> tags
    match = re.search(r'<define>(.*?)</define>', example['src'])
    label = 'SUPPORTED' if example['label'] == 'Not Hallucination' else 'NOT_SUPPORTED'
    # Return a dictionary with the new key 'word'
    return {'word': match.group(1).strip(), 'label': label, 'claim': example['hyp']}


a_new = a.map(extract_word)
a_new = a_new.remove_columns(['hyp'])

data_dict = DatasetDict()
data_dict['test'] = a_new

data_dict.push_to_hub('lukasellinger/shroom-claim_verification', token=HF_WRITE_TOKEN)
