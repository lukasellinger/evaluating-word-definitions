"""Creates the sqlite3 db for storing german dataset."""
import spacy
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from database.db_retriever import FeverDocDB
from utils.spacy_utils import remove_starting_article

CREATE_GERMAN_DATASET = """
CREATE TABLE IF NOT EXISTS german_dpr_dataset (
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

INSERT_ENTRY = """
INSERT INTO german_dpr_dataset (question, claim, fact, word, context, label)
VALUES (?, ?, ?, ?, ?, ?)
"""

QUESTION_CONVERSION = {"Was ist": "{} ist {}.",
                       "Was bezeichnet man als": "Als {} bezeichnet man {}.",
                       "Was bezeichnet": "{} bezeichnet {}.",
                       "Was bedeutet": "{} bedeuet {}.",
                       "Was macht": "{} macht {}.",
                       "Was kennzeichnet": "{} kennzeichnet {}."
                       }


def create_fact(question_sent, answer_sent):
    """Create a fact sentence out of the question and answer."""
    for key, value in QUESTION_CONVERSION.items():
        if question_sent.startswith(key):
            entity = question_sent[len(key) + 1: -1]
            entity = remove_starting_article(entity, lang='ger')  # remove leading article
            if len(entity.split(' ')) > 2:  # right now we do not want entities longer than 2 words
                continue

            fact_sent = value.format(entity, answer_sent).strip()
            fact_sent = fact_sent[0].upper() + fact_sent[1:]
            return fact_sent, entity


def is_def_question(question_sent):
    """Checks whether the entry has a question which asks for a definition of a word."""
    if question_sent.startswith(tuple(QUESTION_CONVERSION.keys())):
        if question_sent.startswith('Was ist'):
            question_tokens = nlp(question_sent)

            word = question_tokens[2]
            if word.pos_ == 'DET':
                word = word.head
            if word.dep_ == 'compound':
                word = word.head
            for child in word.children:
                if child.pos_ == 'ADJ':
                    return False

            if question_sent.endswith(f'{word}?'):
                return True
        else:
            return True
    return False


with FeverDocDB() as db:
    db.write(CREATE_GERMAN_DATASET)

nlp = spacy.load("de_core_news_lg")

dataset_dpr = load_dataset("deepset/germandpr")
dataset_dpr_cc = concatenate_datasets([dataset_dpr['train'], dataset_dpr['test']])
filtered_dataset_dpr = dataset_dpr_cc.filter(lambda i: is_def_question(i['question'].strip()))

with FeverDocDB() as db:
    for entry in tqdm(filtered_dataset_dpr, desc='Inserting Entry'):
        question = entry['question']
        answer = entry['answers'][0]  # always only 1 answer
        if output := create_fact(entry['question'].strip(), answer.strip(' .')):
            fact, entity = output
        else:
            continue

        for pos_ctx in entry['positive_ctxs']['text']:
            label = 'SUPPORTED'
            db.write(INSERT_ENTRY, (question, answer, fact, entity, pos_ctx, label))

        # for neg_ctx in entry['negative_ctxs']['text']:
        #    label = 'REFUTED'
        #   db.write(INSERT_ENTRY, (question, answer, fact, entity, neg_ctx, label))

        # for neg_ctx in entry['hard_negative_ctxs']['text']:
        #   label = 'NOT ENOUGH INFO'  # need to be careful. Ctx might also support fact.
        #    db.write(INSERT_ENTRY, (question, answer, fact, entity, neg_ctx, label))
