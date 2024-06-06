from langdetect import detect
from tqdm import tqdm

from config import PROJECT_DIR
from database.db_retriever import FeverDocDB
from reader import JSONReader
from utils.spacy_utils import german_nlp
from utils.utils import remove_non_alphabetic_start_end

CREATE_GER_DATASET = """
CREATE TABLE IF NOT EXISTS german_dataset (
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

INSERT_ENTRY = """
INSERT OR IGNORE INTO german_dataset (word, label, claim, context_sentence)
VALUES (?, ?, ?, ?)
"""

with FeverDocDB() as db:
    db.write(CREATE_GER_DATASET)

table = JSONReader().read(PROJECT_DIR.joinpath('dataset/jan_eval_results_table.json'))
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

        label = "SUPPORTED"
        db.write(INSERT_ENTRY, (word, label, claim, context_sentence))
