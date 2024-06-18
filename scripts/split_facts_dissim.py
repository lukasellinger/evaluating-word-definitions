from collections import defaultdict

from tqdm import tqdm

from database.db_retriever import FeverDocDB
from general_utils.utils import sentence_simplification, process_dissim_sentence


def main(table, fact_table, claim_col):
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
    SELECT DISTINCT id
    FROM {table}
    WHERE {claim_col} = ?;
    """

    with FeverDocDB() as db:
        db.write(CREATE_ATOMIC_FACTS)
        claims = [entry[0] for entry in db.read(f"""SELECT DISTINCT {claim_col} FROM {table}""")]

    output = sentence_simplification(claims)
    stats = defaultdict(int)
    with FeverDocDB() as db:
        for claim in tqdm(output):
            original_claim = claim.get('sentence')
            claim_ids = [claim_id[0] for claim_id in db.read(GET_CLAIM_ID, (original_claim,))]
            assert claim_ids, f'No claim_ids associated with {original_claim} found.'

            splits = claim.get('splits')
            stats[len(splits)] += len(claim_ids)
            for split in splits:
                split = process_dissim_sentence(split)
                for claim_id in claim_ids:
                    db.write(INSERT_FACT, (claim_id, split))
    return stats


if __name__ == "__main__":
    table = 'def_dataset'
    fact_table = 'atomic_facts_fever_short_dissim'
    claim_col = 'short_claim'
    stats = main(table, fact_table, claim_col)
    print(stats)