from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from database.db_retriever import FeverDocDB
from general_utils.spacy_utils import is_german_def_question, create_german_fact


def main(table, dataset_name):
    CREATE_GERMAN_DATASET = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        claim TEXT,
        english_claim TEXT,
        fact TEXT,
        word VARCHAR,
        english_word VARCHAR,
        context TEXT,
        label VARCHAR
    );
    """

    INSERT_ENTRY = f"""
    INSERT INTO {table} (question, claim, fact, word, context, label)
    VALUES (?, ?, ?, ?, ?, ?)
    """

    with FeverDocDB() as db:
        db.write(CREATE_GERMAN_DATASET)

    dataset = load_dataset(dataset_name)
    dataset_dpr_cc = concatenate_datasets([dataset['train'], dataset['test']])
    filtered_dataset = dataset_dpr_cc.filter(lambda i: is_german_def_question(i['question'].strip()))

    with FeverDocDB() as db:
        for entry in tqdm(filtered_dataset, desc='Inserting Entry'):
            question = entry['question']
            answer = entry['answers'][0]  # always only 1 answer
            if output := create_german_fact(entry['question'].strip(), answer.strip(' .')):
                fact, entity = output
            else:
                continue

            # we can only use the postive ctxs
            # as for the others we cannot be sure if we can nevertheless find it.
            for pos_ctx in entry['positive_ctxs']['text']:
                label = 'SUPPORTED'
                db.write(INSERT_ENTRY, (question, answer, fact, entity, pos_ctx, label))


if __name__ == "__main__":
    main('german_dpr_dataset', 'deepset/germandpr')

