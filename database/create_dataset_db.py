"""Creates the sqlite3 db for storing def dataset."""

from tqdm import tqdm

from config import PROJECT_DIR
from database.db_retriever import FeverDocDB
from reader import JSONLineReader

CREATE_DEF_DATASET = """
CREATE TABLE IF NOT EXISTS def_dataset (
    id INTEGER,
    verifiable VARCHAR,
    label VARCHAR,
    claim TEXT,
    evidence_annotation_id INTEGER,
    evidence_id INTEGER,
    evidence_wiki_url VARCHAR,
    evidence_sentence_id INTEGER,
    set_type VARCHAR,
    PRIMARY KEY (id, evidence_annotation_id, evidence_id)
);
"""

INSERT_ENTRY = """
INSERT INTO def_dataset (id, verifiable, label, claim, evidence_annotation_id, evidence_id,
                         evidence_wiki_url, evidence_sentence_id, set_type)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

with FeverDocDB() as db:
    db.write(CREATE_DEF_DATASET)

train_file_path = PROJECT_DIR.joinpath("dataset/def_train.jsonl")
dev_file_path = PROJECT_DIR.joinpath("dataset/def_dev.jsonl")
test_file_path = PROJECT_DIR.joinpath("dataset/def_test.jsonl")
reader = JSONLineReader()

for set_type, path in zip(['train', 'dev', 'test'],
                          [train_file_path, dev_file_path, test_file_path]):
    dataset = reader.read(path)
    with FeverDocDB() as db:
        for entry in tqdm(dataset, desc='Inserting Entry'):
            entry['set_type'] = set_type
            db.write(INSERT_ENTRY, tuple(entry.values()))
