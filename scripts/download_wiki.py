"""Script to download needed wikipedia articles."""
from datasets import DatasetDict, Dataset, load_dataset
from tqdm import tqdm

from config import HF_WRITE_TOKEN
from fetchers.wikipedia import Wikipedia

datasets = {
    'lukasellinger/german_dpr-claim_verification': {
        'dataset': load_dataset('lukasellinger/german_dpr-claim_verification', split='test',
                                download_mode="force_redownload"),
        'lang': 'de'
    },
    'lukasellinger/german_wiktionary-claim_verification-mini': {
        'dataset': load_dataset('lukasellinger/german_wiktionary-claim_verification-mini',
                                split='test', download_mode="force_redownload"),
        'lang': 'de'
    },
    'lukasellinger/german_wiktionary-claim-verification-large': {
        'dataset': load_dataset('lukasellinger/german_wiktionary-claim-verification-large',
                                split='test', download_mode="force_redownload"),
        'lang': 'de'
    },
    'lukasellinger/german-claim_verification': {
        'dataset': load_dataset('lukasellinger/german-claim_verification', split='test',
                                download_mode="force_redownload"),
        'lang': 'de'
    },
    'lukasellinger/squad-claim_verification': {
        'dataset': load_dataset('lukasellinger/squad-claim_verification', split='test',
                                download_mode="force_redownload"),
        'lang': 'en'
    }
}

all_docs = []
processed_words = {}


def download(example):
    """
    Downloads Wikipedia pages (both full and introductory text) for a given word and fallback word.

    This function checks if the word and fallback word combination has already been processed.
    If not, it fetches the full and introductory Wikipedia pages for the word and updates the
    example with the corresponding document search word.

    :param example: A dictionary containing 'word' and optional 'english_word'.
    :return: The updated example dictionary with the 'document_search_word' added.
    """
    word = example['word']
    fallback_word = example.get('english_word', word)

    if f'{word}{fallback_word}' in processed_words:
        example['document_search_word'] = processed_words[f'{word}{fallback_word}']
        return example

    full_docs, _ = wiki.get_pages(word, fallback_word, word_lang, only_intro=False, return_raw=True)
    intro_docs, document_search_word = wiki.get_pages(word, fallback_word, word_lang,
                                                      only_intro=True, return_raw=True)

    example['document_search_word'] = document_search_word
    processed_words[f'{word}{fallback_word}'] = document_search_word

    full_docs = dict(full_docs)
    intro_docs = dict(intro_docs)

    docs = []
    for title in full_docs:
        docs.append({'search_word': document_search_word,
                     'title': title,
                     'raw_full_text': full_docs.get(title),
                     'raw_intro_text': intro_docs.get(title)})
    all_docs.extend(docs)
    return example


wiki = Wikipedia()
for dataset_name, dataset_info in tqdm(datasets.items()):
    dataset = dataset_info['dataset']
    word_lang = dataset_info['lang']
    dataset = dataset.map(download)

    data_dict = DatasetDict()
    data_dict['test'] = dataset

    data_dict.push_to_hub(dataset_name, token=HF_WRITE_TOKEN)

all_docs_dataset = Dataset.from_list(all_docs)
all_docs_dataset = all_docs_dataset.map(lambda examples, idx: {'id': idx + 1}, with_indices=True)
all_docs_dataset.push_to_hub('lukasellinger/wiki_dump_2024-08-13', token=HF_WRITE_TOKEN)
