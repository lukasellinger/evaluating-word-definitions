from datasets import load_dataset

from config import HF_WRITE_TOKEN

fever = load_dataset('lukasellinger/fever_claim_verification_dissim-v1', split='train')
fever = fever.remove_columns('atomic_facts')
fever.push_to_hub('lukasellinger/filtered_fever-claim_verification', token=HF_WRITE_TOKEN)

