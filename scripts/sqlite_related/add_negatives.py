from collections import defaultdict

from datasets import Dataset
from tqdm import tqdm

from config import DB_URL
from database.db_retriever import FeverDocDB
from general_utils.word_replacer import WordReplacer


def main(table, use_antonyms: bool, only_english: bool):
    INSERT_ENTRY = f"""
    INSERT OR IGNORE INTO {table} (word, label, claim)
    VALUES (?, ?, ?)
    """

    dataset = Dataset.from_sql(f"SELECT word, claim FROM {table}", con=DB_URL, cache_dir=None)
    word_replacer = WordReplacer(use_antonyms, only_english=only_english)
    word_set = list(set([entry['word'] for entry in dataset]))
    stats = defaultdict(int)
    with FeverDocDB() as db:
        for entry in tqdm(dataset):
            word = entry.get('word')
            replacement, stat = word_replacer.get_replacement(word, word_set)
            db.write(INSERT_ENTRY, (replacement, 'NOT_SUPPORTED', entry['claim']))
            stats[stat] += 1
    return stats


if __name__ == "__main__":
    #stats = main('german_dataset', True)
    #print(stats)

    #stats = main('german_dpr_dataset', False)
    #print(stats)

    stats = main('squad_dataset', True, only_english=True)
    print(stats)
