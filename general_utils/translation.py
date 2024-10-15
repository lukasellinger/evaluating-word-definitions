from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(self, source_lang: str, dest_lang: str):
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{dest_lang}'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def get_top_n_translations(self, text, num_translations=5, max_length=100,
                               num_beams=20):
        inputs = self.tokenizer(text, return_tensors='pt')

        # Generate translations using beam search
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_translations,
            early_stopping=True
        )

        translations = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return translations

    def get_translation(self, text, max_length=100, num_beams=20):
        return self.get_top_n_translations(text, 1, max_length, num_beams)[0]
