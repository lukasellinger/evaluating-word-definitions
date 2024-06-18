import random
from typing import List, Tuple
from odenet import antonyms_word
from nltk.corpus import wordnet as wn

from general_utils.spacy_utils import is_single_word
from general_utils.translation import Translator


class WordReplacer:
    def __init__(self, use_antonyms: bool = True, only_english: bool = False):
        self.use_antonyms = use_antonyms
        self.only_english = only_english
        self.translator_en_de = Translator(source_lang='en', dest_lang='de')
        self.translator_de_en = Translator(source_lang='de', dest_lang='en')

    def get_replacement(self, word, word_set: List[str], use_antonyms: bool = None) -> Tuple[str, str]:
        if not use_antonyms:
            use_antonyms = self.use_antonyms

        if self.only_english:
            return self.get_english_replacement(word, word_set, use_antonyms)
        else:
            return self.get_german_replacement(word, word_set, use_antonyms)

    def get_german_replacement(self, word, word_set, use_antonyms: bool = None):
        if use_antonyms:
            antonym, origin = self._get_antonym(word)
            if not antonym:
                antonym, origin = self._translate_and_get_antonym(word)
                if not antonym:
                    antonym, origin = self._get_random_word(word, word_set)
        else:
            antonym, origin = self._get_random_word(word, word_set)

        assert antonym is not None
        return antonym, origin

    def get_english_replacement(self, word, word_set, use_antonyms: bool = None):
        if not use_antonyms:
            use_antonyms = self.use_antonyms

        if use_antonyms:
            antonym, origin = self._get_english_antonym(word)
            if not antonym:
                antonym, origin = self._get_random_word(word, word_set)
        else:
            antonym, origin = self._get_random_word(word, word_set)

        assert antonym is not None
        return antonym, origin

    def _get_antonym(self, word: str) -> tuple[str, str] | tuple[None, None]:
        if antonyms := self.get_odenet_antonyms(word):
            return random.choice(antonyms), 'german_antonym'
        return None, None

    def _translate_and_get_antonym(self, word: str) -> tuple[str, str] | tuple[None, None]:
        translated_word = self.translator_de_en.get_translation(word)
        return self._get_english_antonym(translated_word)

    def _get_english_antonym(self, word: str):
        antonyms = self.get_antonyms(word)
        translated_antonyms = [self.translator_en_de.get_translation(antonym) for antonym in
                               antonyms]
        valid_antonyms = [antonym for antonym in translated_antonyms if
                          is_single_word(antonym, lang='de')]
        if valid_antonyms:
            return random.choice(valid_antonyms), 'english_antonym'
        return None, None

    @staticmethod
    def _get_random_word(word: str, word_set: List[str]) -> Tuple[str, str]:
        word_set.remove(word)
        random_word = random.choice(word_set)
        word_set.append(word)
        return random_word, 'word_set'

    @staticmethod
    def get_antonyms(word: str) -> List:
        antonyms = []
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonym = lemma.antonyms()[0].name()
                    antonym = ' '.join(antonym.split('_'))
                    antonyms.append(antonym)
        return antonyms

    @staticmethod
    def get_odenet_antonyms(word: str) -> List:
        antonyms = []
        if pot_antonyms := antonyms_word(word):  # check with odenet
            for _, _, words in pot_antonyms:
                filtered_words = [word for word in words if is_single_word(word, lang='de')]
                antonyms.extend(filtered_words)
        return antonyms
