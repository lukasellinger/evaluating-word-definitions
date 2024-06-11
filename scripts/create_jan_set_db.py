from langdetect import detect
from tqdm import tqdm

from config import PROJECT_DIR
from database.db_retriever import FeverDocDB
from general_utils.reader import JSONReader
from general_utils.spacy_utils import is_single_word
from general_utils.utils import remove_non_alphabetic_start_end

def main(table, json_table):
    CREATE_GER_DATASET = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word VARCHAR,
        english_word VARCHAR,
        label VARCHAR,
        claim TEXT,
        english_claim VARCHAR,
        context_sentence TEXT,
        UNIQUE(word, claim)
        );
    """

    INSERT_ENTRY = f"""
    INSERT OR IGNORE INTO {table} (word, label, claim, context_sentence)
    VALUES (?, ?, ?, ?)
    """

    with FeverDocDB() as db:
        db.write(CREATE_GER_DATASET)

    table = JSONReader().read(json_table)
    data = table.get('data')

    with FeverDocDB() as db:
        for entry in tqdm(data):
            claim = str(entry[2])
            # clean odd seeming definitions
            if "''" in claim or "<sup>" in claim or len(claim) < 5:
                continue

            prompt_input = entry[1]
            context_sentence = prompt_input.get('context_sentence')
            # we only want words in a german context
            if detect(context_sentence) != 'de':
                continue

            word = prompt_input.get('title')
            word = remove_non_alphabetic_start_end(word)

            if not is_single_word(word, lang='de'):  # we only want single words
                continue

            label = "SUPPORTED"
            db.write(INSERT_ENTRY, (word, label, claim, context_sentence))


if __name__ == "__main__":
    table = 'german_dataset'
    json_table = PROJECT_DIR.joinpath('dataset/jan_eval_results_table.json')
    main(table, json_table)
