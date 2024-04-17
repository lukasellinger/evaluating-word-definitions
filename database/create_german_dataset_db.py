"""Creates the sqlite3 db for storing german dataset."""
import spacy
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from database.db_retriever import FeverDocDB

CREATE_GERMAN_DATASET = """
CREATE TABLE IF NOT EXISTS german_dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT,
    answer TEXT,
    fact TEXT
);
"""

INSERT_ENTRY = """
INSERT INTO german_dataset (question, answer, fact)
VALUES (?, ?, ?)
"""

QUESTION_CONVERSION = {"Was ist": "{} ist {}.",
                       "Was bezeichnet man als": "Als {} bezeichnet man {}.",
                       "Was bezeichnet": "{} bezeichnet {}.",
                       "Was bedeutet": "{} bedeuet {}.",
                       "Was macht": "{} macht {}.",
                       "Was kennzeichnet": "{} kennzeichnet {}."
                       }


def create_fact(entry):
    """Create a fact sentence out of the question and answer."""
    question = entry['question'].strip()
    for key, value in QUESTION_CONVERSION.items():
        if question.startswith(key):
            entity = question[len(key) + 1: -1]
            answer = entry['answers'][0].strip(' .')
            entry['fact'] = value.format(entity, answer).strip()
            entry['fact'] = entry['fact'][0].upper() + entry['fact'][1:]
            return entry
    return entry


def is_german_def_question(entry):
    """Checks whether the entry has a question which asks for a definition of an word"""
    question = str(entry['question']).strip()
    if question.startswith(tuple(QUESTION_CONVERSION.keys())):
        if question.startswith('Was ist'):
            question_tokens = nlp(question)

            word = question_tokens[2]
            if word.pos_ == 'DET':
                word = word.head
            if word.dep_ == 'compound':
                word = word.head
            for child in word.children:
                if child.pos_ == 'ADJ':
                    return False

            if question.endswith(f'{word}?'):
                return True
        else:
            return True
    return False


with FeverDocDB() as db:
    db.write(CREATE_GERMAN_DATASET)

nlp = spacy.load("de_core_news_lg")
dataset = load_dataset("deepset/germandpr")
dataset_cc = concatenate_datasets([dataset['train'], dataset['test']])
filtered_dataset = dataset_cc.filter(lambda entry: is_german_def_question(entry))

with FeverDocDB() as db:
    for entry in tqdm(filtered_dataset, desc='Inserting Entry'):
        question = entry['question']
        answer = entry['answers'][0]  # answers always only 1 answer
        fact = create_fact(entry)['fact']
        db.write(INSERT_ENTRY, (question, answer, fact))
