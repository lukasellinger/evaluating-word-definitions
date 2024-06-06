from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate_to_english(text, model, tokenizer):
    # Eingangssprache auf Deutsch setzen
    tokenizer.src_lang = "de_DE"
    # Tokenisieren vom Eingabetext
    encoded_text = tokenizer(text, return_tensors="pt")
    # Übersetztes Token generieren
    num_translations = 20
    max_length = 50
    num_beams = 50
    outputs = model.generate(
        **encoded_text,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_translations,
        early_stopping=True
    )

    # Dekodieren der generierten Tokens, um den übersetzten Text zu erhalten
    translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return translations

print(translate_to_english("Freundschaftsspiel", model, tokenizer))