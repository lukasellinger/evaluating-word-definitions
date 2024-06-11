from typing import List

from odenet import antonyms_word
from nltk.corpus import wordnet as wn

from utils.spacy_utils import is_single_word


def get_antonyms(word: str) -> List:
    antonyms = []
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return antonyms

word = 'machen'
print(antonyms_word('lachen'))
print(get_antonyms('make'))


antonyms = []
for _, _, words in antonyms:
    print('hiu')


word = 'phonematisches Orthographieprinzip'
print(is_single_word(word, lang='de'))