from datasets import Dataset
from tqdm import tqdm

from config import DB_URL
from database.db_retriever import FeverDocDB
from spacy_utils import get_words_before_root, get_first_compound_or_word
from utils import title_to_db_page

UPDATE = """
UPDATE def_dataset
SET evidence_wiki_url = ?
WHERE id = ?
"""

EXIST_WIKI_PAGE = """
SELECT document_id
FROM documents
WHERE document_id = ?
"""

with FeverDocDB() as db:
    results = Dataset.from_sql("""
                                SELECT DISTINCT id, claim, evidence_wiki_url
                                FROM def_dataset
                                WHERE label = 'NOT ENOUGH INFO'""",
                               con=DB_URL)
    for entry in tqdm(results, desc='Updating Entry'):
        db.write(UPDATE, (None, entry['id']))
        claim = entry['claim']
        word = get_words_before_root(claim)
        page = title_to_db_page(word)
        result = db.read(EXIST_WIKI_PAGE, (page,))
        document_id = page if result else None
        entry['evidence_wiki_url'] = document_id if document_id else None
        db.write(UPDATE, (entry['evidence_wiki_url'], entry['id']))
