from collections import defaultdict

from tqdm import tqdm

from database.db_retriever import FeverDocDB
from reader import JSONReader


CREATE_ATOMIC_FACTS = """
CREATE TABLE IF NOT EXISTS atomic_facts_german_dpr (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    fact TEXT
    );  
"""

INSERT_FACT = """
INSERT INTO atomic_facts_german_dpr (claim_id, fact)
VALUES (?, ?);
"""

GET_CLAIM_ID = """
SELECT id
FROM german_dpr_dataset
WHERE english_claim = ?;
"""

with FeverDocDB() as db:
    db.write(CREATE_ATOMIC_FACTS)

output = JSONReader().read('splits_german_dpr.json').get('sentences')
stats = defaultdict(int)
with FeverDocDB() as db:
    for claim in tqdm(output):
        original_claim = claim.get('originalSentence')
        claim_ids = [claim_id[0] for claim_id in db.read(GET_CLAIM_ID, (original_claim,))]
        assert claim_ids, f'No claim_ids associated with {original_claim} found.'
        splits = claim.get('elementMap')
        stats[len(splits)] += len(claim_ids)

        for split in splits.values():
            fact = split.get('text')
            for claim_id in claim_ids:
                db.write(INSERT_FACT, (claim_id, fact))
print(stats)
