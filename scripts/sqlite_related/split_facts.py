from collections import defaultdict

from tqdm import tqdm

from database.db_retriever import FeverDocDB
from more_itertools import chunked

from pipeline_module.claim_splitter import T5SplitRephraseSplitter


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
        claims = [f'{entry[0]}: {entry[1]}' for entry in db.read(f"""SELECT DISTINCT {word_col}, {claim_col} FROM {table}""")]

    splitter = T5SplitRephraseSplitter()  # MixtralSplitter / DisSimSplitter
    stats = defaultdict(int)
    for batch in tqdm(chunked(claims, 4)):
        output = splitter(batch)

        with FeverDocDB() as db:
            for claim in output:
                original_claim = ': '.join(claim.get('sentence').split(': ')[1:])
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
    fact_table = 'atomic_facts_german_dpr_t5'
    claim_col = 'english_claim'
    word_col = 'english_word'
    stats = main(table, fact_table, claim_col)
    print(stats)
