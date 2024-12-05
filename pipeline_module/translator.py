"""Module for translators."""
from abc import ABC, abstractmethod
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator(ABC):
    """
    Abstract base class for a Translator. Defines the structure for translating words and texts.
    """

    def __call__(self, batch: List[dict]):
        return self.translate_word_text_batch(batch)

    @abstractmethod
    def translate_word_text(self, word: str, text: str) -> dict:
        """
         Translates a single word and its associated text.

         :param word: The word to translate.
         :param text: The text associated with the word.
         :return: Dict containing the translated word and text.
         """

    @abstractmethod
    def translate_word_text_batch(self, batch: List[dict]) -> List[dict]:
        """
        Translates a batch of words and their associated texts.

        :param batch: A batch of dictionaries, each containing 'word' and 'text'.
        :return: The translated batch as a list of dictionaries.
        """

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """
        Translates a given text.

        :param text: The text to translate.
        :return: The translated text.
        """

    @abstractmethod
    def translate_batch(self, batch: List[str], num_translations: int = 5) -> List[List[str]]:
        """
        Translates a batch of strings (texts).

        :param batch: A list of texts to translate.
        :param num_translations: How many translations to return
        :return: The translated batch.
        """

    @abstractmethod
    def translate_claim_batch(self, batch: list[dict]) -> list[dict]:
        """
        Translates a batch of claims.

        :param batch: A batch of dictionaries, each containing 'text'.
        :return: The translated batch as a list of dictionaries.
        """


class OpusMTTranslator(Translator):
    """Helsinki-NLP/opus-mt Translator"""

    def __init__(self, source_lang: str = 'de', dest_lang: str = 'en'):
        self.model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

    def load_model(self):
        """Load the machine learning model for translation, if not already loaded."""
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def unload_model(self):
        """Unload the machine learning model and free up GPU resources."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def translate_word_text(self, word: str, text: str) -> dict:
        return self.translate_word_text_batch([{'word': word, 'text': text}])[0]

    def translate_word_text_batch(self, batch: List[dict]):
        if not self.model:
            self.load_model()

        batch_translations = self.translate_batch([f"{entry.get('word')}: {entry.get('text')}"
                                                   for entry in batch])

        translated_batch, fallback_needed = [], []
        for entry, batch_entry in zip(batch, batch_translations):
            translation_split = next(
                (translation.split(': ') for translation in batch_entry if ': ' in translation),
                None)

            if translation_split:
                translated_batch.append({'word': translation_split[0], 'text': ': '.join(
                    translation_split[1:])})
            else:
                fallback_needed.append(entry)

        if fallback_needed:
            word_index = [entry.get('word') for entry in batch]
            words = [entry.get('word') for entry in fallback_needed]
            translated_words = self.translate_batch(words)
            translated_texts = self.translate_batch([entry.get('text')
                                                     for entry in fallback_needed],
                                                    num_translations=1)

            for word, translated_word, translated_text in zip(words, translated_words,
                                                              translated_texts):
                translated_batch.insert(word_index.index(word),
                                        {'word': translated_word[0], 'text': translated_text[0]})

        return translated_batch

    def translate_claim_batch(self, batch: list[dict]) -> list[dict]:
        batch_translations = self.translate_batch([f"{entry.get('text')}" for entry in batch], num_translations=1)
        return [{'text': translation[0]} for translation in batch_translations]

    def translate_text(self, text: str) -> str:
        return self.translate_batch([text], num_translations=1)[0][0]

    def translate_batch(self, batch: List[str], num_translations: int = 5) -> List[List[str]]:
        return self.get_top_n_translations(batch, num_translations=num_translations)

    def get_top_n_translations(self, batch: List[str], num_translations: int = 5,
                               max_length: int = 100, num_beams: int = 20) -> List[List[str]]:
        """
        Retrieves the top N translations for a batch of texts.

        :param batch: A list of texts to translate.
        :param num_translations: The number of translations to return for each text (default: 5).
        :param max_length: The maximum length of the translation (default: 100 tokens).
        :param num_beams: The number of beams for beam search (default: 20).
        :return: A list of lists containing the top N translations for each input text.
        """
        inputs = self.tokenizer(batch, return_tensors='pt', padding=True).to(self.device)

        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_translations,
            early_stopping=True
        )

        translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [translations[i:i + num_translations] for i in
                range(0, len(translations), num_translations)]


if __name__ == "__main__":
    translator = OpusMTTranslator()
    print(translator([
        {'word': 'Apfel', 'text': 'Ich mag gerne Essen.'},
        {'word': 'Banane', 'text': 'Ich mag gerne Kiwi.'}
    ]))
