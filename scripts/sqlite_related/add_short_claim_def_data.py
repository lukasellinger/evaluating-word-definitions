"""Add short claim to def data."""
from tqdm import tqdm

from database.db_retriever import FeverDocDB
from general_utils.spacy_utils import get_words_after_root

UPDATE_TRANSLATION = """
UPDATE def_dataset
SET short_claim = ?
WHERE id = ? and claim=?;
"""

GET = """
select distinct dd.id, dd.claim
from def_dataset dd
"""

with FeverDocDB() as db:
    claims = db.read(GET)
    for claim_id, claim in tqdm(claims):
        short_claim = get_words_after_root(claim)
        db.write(UPDATE_TRANSLATION, (short_claim, claim_id, claim))
