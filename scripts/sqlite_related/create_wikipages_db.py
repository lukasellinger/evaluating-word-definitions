"""Creates the sqlite3 db for storing wiki-data."""

import os

from tqdm import tqdm

from config import PROJECT_DIR
from database.db_retriever import FeverDocDB
from general_utils.reader import JSONLineReader

CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id VARCHAR,
    text TEXT,
    lines TEXT,
    UNIQUE(document_id)
);
"""

INSERT_DOCUMENT = """
INSERT INTO documents (document_id, text, lines) VALUES (?, ?, ?)
"""

with FeverDocDB() as db:
    db.write(CREATE_DOCUMENTS)

wiki_pages_dir = PROJECT_DIR.joinpath('data/wiki-pages')
wiki_pages = os.listdir(wiki_pages_dir)
reader = JSONLineReader()

for block in tqdm(wiki_pages, desc='Processing Block'):
    block_data = reader.read(wiki_pages_dir.joinpath(block))

    with FeverDocDB() as db:
        for line in block_data:
            db.write(INSERT_DOCUMENT, (line.get('id'), line.get('text'), line.get('lines')))
