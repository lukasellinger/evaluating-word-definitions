from datasets import load_dataset
from tqdm import tqdm

from config import HF_WRITE_TOKEN
from database.db_retriever import FeverDocDB
from fetchers.wikipedia import Wikipedia


def main(table, dataset_name, word_lang):
    CREATE_DATASET = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        search_word VARCHAR,
        title VARCHAR UNIQUE,
        raw_full_text TEXT,
        raw_intro_text TEXT
    );
    """

    INSERT_ENTRY = f"""
    INSERT OR IGNORE INTO {table} (search_word, title, raw_full_text, raw_intro_text)
    VALUES (?, ?, ?, ?)
    """

    with FeverDocDB() as db:
        db.write(CREATE_DATASET)

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

    dataset = load_dataset(dataset_name).get('train')
    wiki = Wikipedia()
    dataset = dataset.map(download)
    dataset.push_to_hub(dataset_name, private=True, token=HF_WRITE_TOKEN)

    with FeverDocDB() as db:
        for doc in all_docs:
            db.write(INSERT_ENTRY, (doc['search_word'], doc['title'],
                                    doc['raw_full_text'], doc['raw_intro_text']))

if __name__ == "__main__":
    main('wiki_test_documents', 'lukasellinger/german_dpr_claim_verification_dissim-v1', word_lang='de')
    main('wiki_test_documents', 'lukasellinger/german_claim_verification_dissim-v1', word_lang='de')
    main('wiki_test_documents', 'lukasellinger/squad_claim_verification_dissim-v1', word_lang='en')


