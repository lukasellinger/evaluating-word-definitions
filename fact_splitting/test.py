from database.db_retriever import FeverDocDB
from reader import LineReader

reader = LineReader()

with FeverDocDB() as db:
    entries = db.read("""select distinct german_dataset.english_claim
from german_dataset""")

lines = [entry[0] for entry in entries]
reader.write('input.txt', lines)