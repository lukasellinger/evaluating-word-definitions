from transformers import pipeline

test = 'Freundschaftsspiel ist eine sportliche Spielpaarung, die in keine offizielle Wertung einfließt.'

pipe_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

print(pipe_to_en(test))