from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Translator(ABC):
    def __call__(self, batch: List[Dict]):
        return self.translate_word_text_batch(batch)

    @abstractmethod
    def translate_word_text(self, word: str, text: str):
        pass

    @abstractmethod
    def translate_word_text_batch(self, batch: List[Dict]):
        pass

    @abstractmethod
    def translate_text(self, text: str):
        pass

    @abstractmethod
    def translate_batch(self, batch: List[str]):
        pass


class OpusMTTranslator(Translator):
    def __init__(self, source_lang: str = 'de', dest_lang: str = 'en'):
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)

    def translate_word_text(self, word: str, text: str):
        return self.translate_word_text_batch([{'word': word, 'text': text}])

    def translate_word_text_batch(self, batch: List[Dict]):
        connected_batch = [f"{entry.get('word')}: {entry.get('text')}" for entry in batch]
        batch_translations = self.translate_batch(connected_batch)

        translated_batch = []
        fallback_needed = []

        for entry, batch_entry in zip(batch, batch_translations):
            translation_split = next(
                (translation.split(': ') for translation in batch_entry if ': ' in translation),
                None)

            if translation_split:
                translated_word, translated_text = translation_split[0], ': '.join(
                    translation_split[1:])
                translated_batch.append({'word': translated_word, 'text': translated_text})
            else:
                fallback_needed.append(entry)

        if fallback_needed:
            word_index = [entry.get('word') for entry in batch]
            words = [entry.get('word') for entry in fallback_needed]
            texts = [entry.get('text') for entry in fallback_needed]

            translated_words = self.translate_batch(words, num_translations=1)
            translated_texts = self.translate_batch(texts, num_translations=1)

            for word, translated_word, translated_text in zip(words, translated_words,
                                                              translated_texts):
                translated_batch.insert(word_index.index(word),
                                        {'word': translated_word, 'text': translated_text})

        return translated_batch

    def translate_text(self, text: str):
        return self.translate_batch([text], num_translations=1)[0][0]

    def translate_batch(self, batch: List[str], num_translations: int = 5):
        return self.get_top_n_translations(batch, num_translations=num_translations)

    def get_top_n_translations(self, batch: List[str], num_translations: int = 5,
                               max_length: int = 100, num_beams: int = 20):
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

    print(translator.translate_word_text(word='Apfel', text='Banane'))
    print(translator.translate_text(text='Hallo wie gehts'))
