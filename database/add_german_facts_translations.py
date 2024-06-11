from tqdm import tqdm

from config import HF_READ_TOKENS
from database.db_retriever import FeverDocDB
from fact_extractor import FactExtractor
from translation.translation import Translator

CREATE_ATOMIC_FACTS = """
CREATE TABLE IF NOT EXISTS atomic_facts_german (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    fact TEXT
    );  
"""

CREATE_EXPLANATIONS = """
CREATE TABLE IF NOT EXISTS atomic_facts_german_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    explanation TEXT);
"""

INSERT_FACT = """
INSERT INTO atomic_facts_german (claim_id, fact)
VALUES (?, ?);
"""

UPDATE_TRANSLATION = """
UPDATE german_dataset
SET english_claim = ?, english_word = ?
WHERE id = ?;
"""

INSERT_EXPLANATION = """
INSERT INTO atomic_facts_german_explanations (claim_id, explanation)
VALUES (?, ?);
"""

with FeverDocDB() as db:
    db.write(CREATE_ATOMIC_FACTS)
    db.write(CREATE_EXPLANATIONS)
    claims = db.read("""SELECT DISTINCT dd.id, dd.word, dd.claim
                        FROM german_dataset dd
                        LEFT JOIN atomic_facts_german af on af.claim_id = dd.id
                        WHERE af.id is NULL""")

extractor = FactExtractor(hf_token=HF_READ_TOKENS[1])
translator = Translator(source_lang='de', dest_lang='en')

with FeverDocDB() as db:
    for claim_id, word, claim in tqdm(claims):
        # translate word and claim at once to get better translation for word
        combined = f'{word}: {claim}'
        translations = translator.get_top_n_translations(combined)
        translation_split = None
        for translation in translations:
            split = translation.split(': ')
            if len(split) > 1:
                translation_split = split
                break

        if translation_split:
            english_word = translation_split[0]
            english_claim = ': '.join(translation_split[1:])
        else:  # fallback as in rare cases translation is not in this format
            english_word = translator.get_translation(word)
            english_claim = translator.get_translation(claim)

        # english_facts = extractor.get_atomic_facts(translation)

        db.write(UPDATE_TRANSLATION, (english_claim, english_word, claim_id))
        
        #if len(claim) <= 30:  # these are not split into atomic facts
        #    continue
            
        #for fact in english_facts.get('facts'):
        #    db.write(INSERT_FACT, (claim_id, fact))

        #if explanation := english_facts.get('explanation'):
        #    db.write(INSERT_EXPLANATION, (claim_id, explanation))
