"""Create squad dataset."""
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from database.db_retriever import FeverDocDB
from general_utils.spacy_utils import create_english_fact, split_into_sentences
from general_utils.utils import find_substring_in_list


def main(table, dataset_name):
    """Main for different tables."""
    create_dataset = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        claim TEXT,
        fact TEXT,
        word VARCHAR,
        wiki_page VARCHAR,
        context TEXT,
        start_sentence INTEGER,
        label VARCHAR
    );
    """

    insert_entry = f"""
    INSERT INTO {table} (question, claim, fact, word, wiki_page, context, start_sentence, label)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    dataset = load_dataset(dataset_name)
    dataset_dpr_cc = concatenate_datasets([dataset['train'], dataset['validation']])

    with FeverDocDB() as db:
        db.write(create_dataset)

        for entry in tqdm(dataset_dpr_cc, desc='Inserting Entry'):
            question = entry['question']
            context = entry['context']
            answer = entry['answers'].get('text')[0]  # most of the time 1 answer
            answer_start = entry['answers'].get('answer_start')[0]

            if output := create_english_fact(question.strip(), answer.strip(' .')):
                fact, entity = output

                sentences = split_into_sentences(context)
                sentence_after_start = split_into_sentences(context[answer_start:])
                if len(sentence_after_start) == 1:  # last sentence of context
                    start_sentence = len(sentences) - 1
                else:
                    try:
                        start_sentence = find_substring_in_list(sentences,
                                                                sentence_after_start[1]) - 1
                    except:
                        print(entry.get('title'))
                        print(entry.get('question'))
                        break

                wiki_page = entry.get('title')
                label = 'SUPPORTED'
                db.write(insert_entry, (question, answer, fact, entity, wiki_page, context,
                                        start_sentence, label))


if __name__ == "__main__":
    main('squad_dataset', 'rajpurkar/squad')
