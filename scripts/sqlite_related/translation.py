import requests
from transformers import pipeline

from config import HF_READ_TOKENS
from database.db_retriever import FeverDocDB

CREATE_CLAIM_TRANSLATIONS = """
CREATE TABLE IF NOT EXISTS claim_translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER,
    translation TEXT
    );  
"""

INSERT_TRANSLATION = """
INSERT INTO claim_translations (claim_id, translation)
VALUES (?, ?);
"""

API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-de"
headers = {"Authorization": f"Bearer {HF_READ_TOKENS[0]}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


with FeverDocDB() as db:
    db.write(CREATE_CLAIM_TRANSLATIONS)
    claims = db.read("""SELECT DISTINCT dd.id, dd.claim 
                        FROM def_dataset dd
                        LEFT JOIN claim_translations ct on dd.id = ct.claim_id 
                        WHERE ct.id is NULL AND length(claim) <= 30""")

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
for claim_id, claim in claims:
    output = pipe(claim)
    answer = output[0].get('translation_text')
    if answer:
        with FeverDocDB() as db:
            db.write(INSERT_TRANSLATION, (claim_id, answer))
