"""Split facts for datasets in database."""
from collections import defaultdict

from more_itertools import chunked
from tqdm import tqdm

from database.db_retriever import FeverDocDB
from pipeline_module.claim_splitter import T5SplitRephraseSplitter


def main(table, fact_table, claim_col, word_col):
    """Main for different tables."""
    create_atomic_facts = f"""
    CREATE TABLE IF NOT EXISTS {fact_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        fact TEXT
        );  
    """

    insert_fact = f"""
    INSERT INTO {fact_table} (claim_id, fact)
    VALUES (?, ?);
    """

    get_claim_id = f"""
    SELECT DISTINCT id
    FROM {table}
    WHERE {claim_col} = ?;
    """

    with FeverDocDB() as db:
        db.write(create_atomic_facts)
        claims = [f'{entry[0]}: {entry[1]}'
                  for entry in db.read(f"""SELECT DISTINCT {word_col}, {claim_col} FROM {table}""")]

    splitter = T5SplitRephraseSplitter()  # MixtralSplitter / DisSimSplitter
    stats = defaultdict(int)
    for batch in tqdm(chunked(claims, 4)):
        output = splitter(batch)

        with FeverDocDB() as db:
            for claim in output:
                original_claim = ': '.join(claim.get('sentence').split(': ')[1:])
                claim_ids = [claim_id[0]
                             for claim_id in db.read(get_claim_id, (original_claim,))]
                assert claim_ids, f'No claim_ids associated with {original_claim} found.'

                splits = claim.get('splits')
                stats[len(splits)] += len(claim_ids)
                for split in splits:
                    for claim_id in claim_ids:
                        db.write(insert_fact, (claim_id, split))
    return stats


if __name__ == "__main__":
    TABLE = 'german_dpr_dataset'
    FACT_TABLE = 'atomic_facts_german_dpr_t5'
    CLAIM_COL = 'english_claim'
    WORD_COL = 'english_word'
    fact_stats = main(TABLE, FACT_TABLE, CLAIM_COL, WORD_COL)
    print(fact_stats)
