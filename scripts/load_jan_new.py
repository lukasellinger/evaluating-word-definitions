from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from langdetect import detect

from config import PROJECT_DIR, HF_WRITE_TOKEN
from general_utils.spacy_utils import is_single_word

data_files = {
    #'train': str(PROJECT_DIR / 'dataset/jan/0_10_words/train.parquet'),
    #'val': str(PROJECT_DIR / 'dataset/jan/0_10_words/val.parquet'),
    'test': str(PROJECT_DIR / 'dataset/jan/0_10_words/test.parquet')
}

# Load the dataset
dataset = load_dataset('parquet', data_files=data_files)
combined_dataset = concatenate_datasets([dataset['test']])
combined_dataset = combined_dataset.shuffle(seed=42)
combined_dataset = combined_dataset.filter(lambda example: is_single_word(example['title'], lang='de'))

# Select 100 samples
sample_dataset = combined_dataset.select(range(100))
sample_dataset = sample_dataset.rename_column('title', 'word')
sample_dataset = sample_dataset.rename_column('wiktionary_gt', 'claim')
sample_dataset = sample_dataset.rename_column('gt', 'llama_claim')

sample_dataset = sample_dataset.map(lambda examples, idx: {'id': idx + 1, 'label': 'SUPPORTED'}, with_indices=True)

df = sample_dataset.to_pandas()

# Reorder columns
columns_order = ['id', 'word', 'claim', 'label', 'llama_claim', 'context_sentence']
df = df[columns_order]

# Convert back to Hugging Face dataset
sample_dataset = Dataset.from_pandas(df)

data_dict = DatasetDict()
data_dict['test'] = sample_dataset

data_dict.push_to_hub('lukasellinger/german_wiktionary-claim_verification-mini', private=True, token=HF_WRITE_TOKEN)

print('hi')