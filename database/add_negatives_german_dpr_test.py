import random

from datasets import Dataset
from tqdm import tqdm

from config import DB_URL
from database.db_retriever import FeverDocDB

INSERT_ENTRY = """
INSERT OR IGNORE INTO german_dpr_dataset (word, label, claim)
VALUES (?, ?, ?)
"""

dataset = Dataset.from_sql("SELECT word, claim FROM german_dpr_dataset WHERE 35=35", con=DB_URL)
words = list(set([entry['word'] for entry in dataset]))

with FeverDocDB() as db:
    for entry in tqdm(dataset):
        word = entry.get('word')
        claim = entry.get('claim')
        words.remove(word)
        random_word = random.choice(words)
        words.append(word)
        db.write(INSERT_ENTRY, (random_word, 'NOT_SUPPORTED', claim))
