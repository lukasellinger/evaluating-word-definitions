from tqdm import tqdm

from config import HF_READ_TOKENS
from database.db_retriever import FeverDocDB
from general_utils.fact_extractor import FactExtractor
from general_utils.translation import Translator


def main(table, fact_table, explanation_table):
    CREATE_ATOMIC_FACTS = f"""
    CREATE TABLE IF NOT EXISTS {fact_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        fact TEXT
        );  
    """

    CREATE_EXPLANATIONS = f"""
    CREATE TABLE IF NOT EXISTS {explanation_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        explanation TEXT);
    """

    INSERT_FACT = f"""
    INSERT INTO {fact_table} (claim_id, fact)
    VALUES (?, ?);
    """

    UPDATE_TRANSLATION = f"""
    UPDATE {table}
    SET english_claim = ?, english_word = ?
    WHERE id = ?;
    """

    INSERT_EXPLANATION = f"""
    INSERT INTO {explanation_table} (claim_id, explanation)
    VALUES (?, ?);
    """

    with FeverDocDB() as db:
        db.write(CREATE_ATOMIC_FACTS)
        db.write(CREATE_EXPLANATIONS)
        claims = db.read(f"""SELECT DISTINCT dd.id, dd.word, dd.claim
                            FROM {table} dd
                            LEFT JOIN {fact_table} af on af.claim_id = dd.id
                            WHERE af.id is NULL""")

    extractor = FactExtractor(hf_token=HF_READ_TOKENS[0])
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

            db.write(UPDATE_TRANSLATION, (english_claim, english_word, claim_id))

            if len(claim) <= 30:  # these are not split into atomic facts
                continue

            english_facts = extractor.get_atomic_facts(f'{english_word}: {english_claim}')

            for fact in english_facts.get('facts'):
                db.write(INSERT_FACT, (claim_id, fact))

            if explanation := english_facts.get('explanation'):
                db.write(INSERT_EXPLANATION, (claim_id, explanation))


if __name__ == "__main__":
    table = 'german_dataset'
    fact_table = 'atomic_facts_german_mixtral'
    explanation_table = 'atomic_facts_german_explanation_mixtral'
    main(table, fact_table, explanation_table)
