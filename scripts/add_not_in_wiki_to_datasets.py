from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm

from config import HF_WRITE_TOKEN
from fetchers.wikipedia import Wikipedia


def main(dataset_name, wiki):
    dataset = load_dataset(dataset_name).get('train')

    not_in_wiki = defaultdict(int)
    in_wiki_col = []
    for entry in tqdm(dataset):
        texts, _ = wiki.get_pages('', '', only_intro=True, return_raw=True,
                                  search_word=entry['document_search_word'])
        if not texts:
            texts_long, _ = wiki.get_pages('', '', only_intro=False, return_raw=True,
                                           search_word=entry['document_search_word'])
            if not texts_long:
                in_wiki_col.append('No')
                not_in_wiki['No'] += 1
            else:
                in_wiki_col.append('No intro')
                not_in_wiki['No intro'] += 1
        else:
            in_wiki_col.append('Yes')
            not_in_wiki['Yes'] += 1
    dataset = dataset.add_column('in_wiki', in_wiki_col)
    dataset.push_to_hub(dataset_name, private=True, token=HF_WRITE_TOKEN)
    return not_in_wiki


if __name__ == "__main__":
    dataset_names = ["lukasellinger/german_dpr_claim_verification_dissim-v1",
                     "lukasellinger/german_claim_verification_dissim-v1",
                     "lukasellinger/squad_claim_verification_dissim-v1"]
    offline_wiki = 'lukasellinger/wiki_dump_2024-07-08'
    wiki = Wikipedia(use_dataset=offline_wiki)
    for dataset_name in dataset_names:
        stats = main(dataset_name, wiki)
        print(f'{dataset_name} - not in Wiki: {stats}')