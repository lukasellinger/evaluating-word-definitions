from collections import defaultdict

from datasets import load_dataset

from config import HF_WRITE_TOKEN
from general_utils.utils import sentence_simplification, process_sentence


def main(dataset_name):
    stats = defaultdict(int)
    dataset = load_dataset(dataset_name).get('train')
    claims = dataset['connected_claim']
    simplified_claims = sentence_simplification(claims)

    add_to_dataset = []
    for simple_claim in simplified_claims:
        splits = simple_claim.get('splits')
        splits = [process_sentence(split) for split in splits]
        stats[len(splits)] += 1
        add_to_dataset.append('--;--'.join(splits))

    # facts_old = dataset['atomic_facts']
    dataset = dataset.remove_columns(['atomic_facts'])
    dataset = dataset.add_column('atomic_facts', add_to_dataset)
    #dataset = dataset.add_column('atomic_facts_old', facts_old)
    dataset.push_to_hub(dataset_name, private=True, token=HF_WRITE_TOKEN)
    return stats

if __name__ == "__main__":
    print(main('lukasellinger/german_dpr_claim_verification_dissim-v1'))
    print(main('lukasellinger/squad_claim_verification_dissim-v1'))
    print(main('lukasellinger/german_claim_verification_dissim-v1'))

