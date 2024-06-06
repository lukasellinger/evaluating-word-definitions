from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
model_name = 'Helsinki-NLP/opus-mt-de-en'  # Example model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Function to get top N translations
def get_top_n_translations(text, model, tokenizer, num_translations=20, max_length=50, num_beams=50):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')

    # Generate translations using beam search
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_translations,
        early_stopping=True
    )

    # Decode the generated translations
    translations = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return translations


# Example text to translate
text = "Angsthase"

# Get top 3 translations
top_translations = get_top_n_translations(text, model, tokenizer)
print("Top Translations:")
for i, translation in enumerate(top_translations, 1):
    print(f"{i}: {translation}")