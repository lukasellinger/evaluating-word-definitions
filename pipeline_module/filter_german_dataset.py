from datasets import load_dataset, DatasetDict

from config import HF_WRITE_TOKEN

dataset = load_dataset('lukasellinger/german-claim_verification', split='test')

def filter_function(example):
    return example['id'] not in [229, 226, 529, 179, 222, 224]

print(len(dataset))
dataset = dataset.filter(filter_function)
print(len(dataset))
data_dict = DatasetDict()
data_dict['test'] = dataset
data_dict.push_to_hub('lukasellinger/german-claim_verification', private=True, token=HF_WRITE_TOKEN)
