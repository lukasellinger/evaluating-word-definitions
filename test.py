import spacy
from deep_translator import PonsTranslator
from deep_translator.exceptions import ElementNotFoundInGetRequest
from langdetect import detect
from odenet import antonyms_word

from database.db_retriever import FeverDocDB



print(detect("They took Charltons gun from his cold, dead hands"))


word = 'Nudossi'
try:
    translated_word = PonsTranslator(source='german', target='english').translate(word)
except ElementNotFoundInGetRequest:
    print('hi')




german_nlp = spacy.load("de_core_news_lg")

with FeverDocDB() as db:
    words = db.read("""SELECT word FROM german_dataset""")

for word in words:
    doc = german_nlp(word[0])
    basic_form = [token.lemma_ for token in doc][0]
    print(antonyms_word(basic_form))


def antonyme_experimentell(original_sentence):
    # ChatGPT wurde als Hilfestellung verwendet
    def translate_to_english(text, model, tokenizer):
        # Eingangssprache auf Deutsch setzen
        tokenizer.src_lang = "de_DE"
        # Tokenisieren vom Eingabetext
        encoded_text = tokenizer(text, return_tensors="pt")
        # Übersetztes Token generieren
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
        # Dekodieren der generierten Tokens, um den übersetzten Text zu erhalten
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

    def translate_to_german(text, model, tokenizer):
        # Eingangssprache auf Englisch setzen
        tokenizer.src_lang = "en_XX"
        # Tokenisieren vom Eingabetext
        encoded_text = tokenizer(text, return_tensors="pt")
        # Übersetztes Token generieren
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"]
        )
        # Dekodieren der generierten Tokens, um den übersetzten Text zu erhalten
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translated_text

    # Funktion um Antonyme mit WordNet zu finden
    # ChatGPT wurde als Hilfestellung verwendet
    def get_antonyms(word):
        antonyms = []
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name())
        return antonyms

    # Funktion um Tokens mit Antonymen zu ersetzen
    def replace_with_antonyms(sentence):
        doc = nlp_en(sentence)
        new_sentence = []
        for token in doc:
            antonyms = get_antonyms(token.text)
            if antonyms:
                new_sentence.append(antonyms[0])  # Erstes Antonym nehmen
            else:
                new_sentence.append(token.text)
        return " ".join(new_sentence)

    # Übersetzen vom Eingabesatz ins Englische
    english_sentence = translate_to_english(original_sentence, model, tokenizer)

    # Satz mit Antonymen erhalten
    modified_sentence = translate_to_german(replace_with_antonyms(english_sentence), model, tokenizer)

    return modified_sentence