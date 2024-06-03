from tqdm import tqdm
from transformers import pipeline

from config import HF_READ_TOKENS
from database.db_retriever import FeverDocDB
from fact_extractor import FactExtractor


CREATE_ATOMIC_FACTS = """
CREATE TABLE IF NOT EXISTS atomic_facts_german_dpr (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    fact TEXT
    );  
"""

CREATE_EXPLANATIONS = """
CREATE TABLE IF NOT EXISTS atomic_facts_german_dpr_explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    explanation TEXT);
"""

INSERT_FACT = """
INSERT INTO atomic_facts_german_dpr (claim_id, fact)
VALUES (?, ?);
"""

INSERT_TRANSLATION = """
INSERT INTO german_dpr_dataset(id, english_claim, english_word)
VALUES (?, ?, ?);
"""

INSERT_EXPLANATION = """
INSERT INTO atomic_facts_german_dpr_explanations (claim_id, explanation)
VALUES (?, ?);
"""

with FeverDocDB() as db:
    db.write(CREATE_ATOMIC_FACTS)
    db.write(CREATE_EXPLANATIONS)
    claims = db.read("""SELECT DISTINCT dd.id, word, dd.word || ': ' || dd.claim as claim
                        FROM german_dpr_dataset dd
                        LEFT JOIN atomic_facts_german_dpr af on af.claim_id = dd.id
                        WHERE af.id is NULL and length(dd.word || ': ' || dd.claim) > 30 and 5=5""")


extractor = FactExtractor(hf_token=HF_READ_TOKENS[2])
pipe_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

with FeverDocDB() as db:
    for claim_id, word, claim in tqdm(claims):
        english_word = pipe_to_en(word)[0].get('translation_text')
        english_claim = pipe_to_en(claim)[0].get('translation_text')
        english_facts = extractor.get_atomic_facts(f'{english_claim}')

        db.write(INSERT_TRANSLATION, (claim_id, english_claim, english_word))

        for fact in english_facts.get('facts'):
            db.write(INSERT_FACT, (claim_id, fact))
        
        if explanation := english_facts.get('explanation'):
            db.write(INSERT_EXPLANATION, (claim_id, explanation))

