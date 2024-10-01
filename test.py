from datasets import load_dataset, tqdm, DatasetDict, Dataset

from config import HF_WRITE_TOKEN
from fetchers.wikipedia import Wikipedia

datasets = {
    'lukasellinger/german_wiktionary-claim_verification-large': {
        'dataset': load_dataset('lukasellinger/german_wiktionary-claim_verification-large', split='test'),
        'lang': 'de'
    },
    'lukasellinger/german_dpr-claim_verification': {
        'dataset': load_dataset('lukasellinger/german_dpr-claim_verification', split='test'),
        'lang': 'de'
    },
    'lukasellinger/german_wiktionary-claim_verification-mini': {
        'dataset': load_dataset('lukasellinger/german_wiktionary-claim_verification-mini', split='test'),
        'lang': 'de'
    },
    'lukasellinger/german-claim_verification': {
        'dataset': load_dataset('lukasellinger/german-claim_verification', split='test'),
        'lang': 'de'
    },
    'lukasellinger/squad-claim_verification': {
        'dataset': load_dataset('lukasellinger/squad-claim_verification', split='test'),
        'lang': 'en'
    },
    'lukasellinger/shroom-claim_verification': {
        'dataset': load_dataset('lukasellinger/shroom-claim_verification', split='test'),
        'lang': 'en'
    }
}


all_docs = []
processed_words = {}

def download(example):
    word = example['word']
    fallback_word = example.get('english_word', word)

    if f'{word}{fallback_word}' in processed_words:
        example['document_search_word'] = processed_words[f'{word}{fallback_word}']
        return example

    full_docs, _ = wiki.get_pages(word, fallback_word, word_lang, only_intro=False, return_raw=True)
    intro_docs, document_search_word = wiki.get_pages(word, fallback_word, word_lang, only_intro=True, return_raw=True)

    example['document_search_word'] = document_search_word
    processed_words[f'{word}{fallback_word}'] = document_search_word

    full_docs = {key: value for key, value in full_docs}
    intro_docs = {key: value for key, value in intro_docs}

    docs = []
    for title in full_docs:
        docs.append({'search_word': document_search_word,
                     'title': title,
                     'raw_full_text': full_docs.get(title),
                     'raw_intro_text': intro_docs.get(title)})
    all_docs.extend(docs)
    return example

for dataset_name, dataset_info in tqdm(datasets.items()):
    dataset = dataset_info['dataset']
    word_lang = dataset_info['lang']
    assert dataset[0].get('word'), f'No word in dataset {dataset_name}'


wiki = Wikipedia()
for dataset_name, dataset_info in tqdm(datasets.items()):
    dataset = dataset_info['dataset']
    word_lang = dataset_info['lang']
    dataset = dataset.map(download)

    data_dict = DatasetDict()
    data_dict['test'] = dataset

    data_dict.push_to_hub(f'{dataset_name}', token=HF_WRITE_TOKEN)


all_docs_dataset = Dataset.from_list(all_docs)
all_docs_dataset = all_docs_dataset.map(lambda examples, idx: {'id': idx + 1}, with_indices=True)
all_docs_dataset.push_to_hub('lukasellinger/wiki_dump_2024-09-27', token=HF_WRITE_TOKEN)
