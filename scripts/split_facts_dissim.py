from collections import defaultdict

from tqdm import tqdm

from database.db_retriever import FeverDocDB
from general_utils.utils import sentence_simplfication


def main(table, fact_table):
    CREATE_ATOMIC_FACTS = f"""
    CREATE TABLE IF NOT EXISTS {fact_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        fact TEXT
        );  
    """

    INSERT_FACT = f"""
    INSERT INTO {fact_table} (claim_id, fact)
    VALUES (?, ?);
    """

    GET_CLAIM_ID = f"""
    SELECT id
    FROM {table}
    WHERE english_claim = ?;
    """

    with FeverDocDB() as db:
        db.write(CREATE_ATOMIC_FACTS)
        claims = [entry[0] for entry in db.read(f"""SELECT DISTINCT english_claim FROM {table}""")]

    output = sentence_simplfication(claims)
    stats = defaultdict(int)
    with FeverDocDB() as db:
        for claim in tqdm(output):
            original_claim = claim.get('sentence')
            claim_ids = [claim_id[0] for claim_id in db.read(GET_CLAIM_ID, (original_claim,))]
            assert claim_ids, f'No claim_ids associated with {original_claim} found.'

            splits = claim.get('splits')
            stats[len(splits)] += len(claim_ids)
            for split in splits:
                for claim_id in claim_ids:
                    db.write(INSERT_FACT, (claim_id, split))
    return stats


if __name__ == "__main__":
    table = 'german_dpr_dataset'
    fact_table = 'atomic_facts_german_dpr'
    stats = main(table, fact_table)
    print(stats)