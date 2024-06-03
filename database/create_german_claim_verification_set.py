import random
from collections import defaultdict
from typing import List

from datasets import Dataset
from odenet import antonyms_word
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from transformers import pipeline

from config import DB_URL
from database.db_retriever import FeverDocDB

INSERT_ENTRY = """
INSERT OR IGNORE INTO german_dataset (word, label, claim)
VALUES (?, ?, ?)
"""


def get_antonyms(word: str) -> List:
    antonyms = []
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms


dataset = Dataset.from_sql("SELECT word, claim FROM german_dataset", con=DB_URL)

pipe_to_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
pipe_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

words = list(set([entry['word'] for entry in dataset]))
stats = defaultdict(int)
with FeverDocDB() as db:
    for entry in tqdm(dataset):
        word = entry['word']

        if antonyms := antonyms_word(word):
            # select word from random context
            context = random.choice(antonyms)
            antonym = random.choice(context[2])
            stats['german_antonym'] += 1
        else:  # translate to english and check with wordnet
            translated_word = pipe_to_en(word)[0].get('translation_text')
            antonyms = get_antonyms(translated_word)
            if antonyms:
                antonym = random.choice(antonyms)
                antonym = pipe_to_de(antonym)[0].get('translation_text')
                stats['english_antonym'] += 1
            else:  # select random word
                words.remove(word)
                antonym = random.choice(words)
                words.append(word)
                stats['random_word'] += 1

        assert antonym is not None
        db.write(INSERT_ENTRY, (antonym, 'NOT_SUPPORTED', entry['claim']))

print(stats)
