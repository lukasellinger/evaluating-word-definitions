from datasets import load_dataset, DatasetDict
from tqdm import tqdm

from config import HF_WRITE_TOKEN
from pipeline_module.claim_splitter import DisSimSplitter


dataset_name = 'lukasellinger/german_wiktionary-claim-verification-large'

splitter = DisSimSplitter()
dataset = load_dataset(dataset_name, split='test', download_mode="force_redownload")

splitted_claims = []
renamed_dataset = dataset.rename_column('connected_claim', 'text')
list_dataset = renamed_dataset.to_list()
batch_size = len(dataset)

for i in tqdm(range(0, len(dataset), batch_size)):
    batch = list_dataset[i:i + batch_size]
    batch = [entry['text'] for entry in batch]
    splitted_batch = splitter(batch)
    splitted_claims.extend('--;--'.join(entry.get('splits')) for entry in splitted_batch)

dataset = dataset.add_column('DisSim_facts', splitted_claims)

data_dict = DatasetDict()
data_dict['test'] = dataset

data_dict.save_to_disk('large_dataset')
data_dict.push_to_hub(dataset_name, token=HF_WRITE_TOKEN)