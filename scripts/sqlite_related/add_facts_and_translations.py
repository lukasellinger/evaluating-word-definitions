"""Script to add facts and translations to dataset in database"""
from tqdm import tqdm

from config import HF_READ_TOKENS
from database.db_retriever import FeverDocDB
from general_utils.translation import Translator
from pipeline_module.claim_splitter import MixtralSplitter


def main(table, fact_table, explanation_table):
    """Main for different tables."""
    create_atomic_facts = f"""
    CREATE TABLE IF NOT EXISTS {fact_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        fact TEXT
        );  
    """

    create_explanations = f"""
    CREATE TABLE IF NOT EXISTS {explanation_table} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        claim_id INTEGER,
        explanation TEXT);
    """

    insert_fact = f"""
    INSERT INTO {fact_table} (claim_id, fact)
    VALUES (?, ?);
    """

    update_translation = f"""
    UPDATE {table}
    SET english_claim = ?, english_word = ?
    WHERE id = ?;
    """

    insert_explanation = f"""
    INSERT INTO {explanation_table} (claim_id, explanation)
    VALUES (?, ?);
    """

    with FeverDocDB() as db:
        db.write(create_atomic_facts)
        db.write(create_explanations)
        claims = db.read(f"""SELECT DISTINCT dd.id, dd.word, dd.claim
                            FROM {table} dd
                            LEFT JOIN {fact_table} af on af.claim_id = dd.id
                            WHERE af.id is NULL""")

    extractor = MixtralSplitter(hf_token=HF_READ_TOKENS[0])
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

            db.write(update_translation, (english_claim, english_word, claim_id))

            if len(claim) <= 30:  # these are not split into atomic facts
                continue

            english_facts = extractor.get_atomic_claims(f'{english_word}: {english_claim}')

            for fact in english_facts.get('splits'):
                db.write(insert_fact, (claim_id, fact))

            if explanation := english_facts.get('explanation'):
                db.write(insert_explanation, (claim_id, explanation))


if __name__ == "__main__":
    TABLE = 'german_dataset'
    FACT_TABLE = 'atomic_facts_german_mixtral'
    EXPLANATION_TABLE = 'atomic_facts_german_explanation_mixtral'
    main(TABLE, FACT_TABLE, EXPLANATION_TABLE)
