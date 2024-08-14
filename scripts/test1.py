from datasets import load_dataset

from config import HF_WRITE_TOKEN

dataset = load_dataset('lukasellinger/german_wiktionary-claim-verification-large',  download_mode="force_redownload")
dataset.push_to_hub('lukasellinger/german_wiktionary-claim_verification-large', token=HF_WRITE_TOKEN)
print(dataset['test'][0])
