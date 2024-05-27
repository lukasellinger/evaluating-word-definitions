from tqdm import tqdm

from database.db_retriever import FeverDocDB
from reader import JSONReader
from utils import remove_non_alphabetic_start_end

CREATE_GER_DATASET = """
CREATE TABLE IF NOT EXISTS german_dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word VARCHAR,
    label VARCHAR,
    claim TEXT,
    UNIQUE(word, claim)
    );
"""

INSERT_ENTRY = """
INSERT OR IGNORE INTO german_dataset (word, label, claim)
VALUES (?, ?, ?)
"""

with FeverDocDB() as db:
    db.write(CREATE_GER_DATASET)

table = JSONReader().read('jan_eval_results_table.json')
data = table.get('data')

with FeverDocDB() as db:
    for entry in tqdm(data):
        word = entry[1].get('context_word')
        word = remove_non_alphabetic_start_end(word)
        label = "SUPPORTED"
        claim = entry[2]
        db.write(INSERT_ENTRY, (word, label, claim))
