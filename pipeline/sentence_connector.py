from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class SentenceConnector(ABC):
    def __call__(self, batch: List[Dict]) -> List[str]:
        return self.connect_batch(batch)

    @abstractmethod
    def connect_word_text(self, word: str, text: str) -> str:
        pass

    @abstractmethod
    def connect_batch(self, batch: List[Dict]) -> List[str]:
        pass


class ColonSentenceConnector(SentenceConnector):
    def __call__(self, batch: List[Dict]) -> List[str]:
        return self.connect_batch(batch)

    def connect_word_text(self, word: str, text: str) -> str:
        return f'{word}: {text}'

    def connect_batch(self, batch: List[Dict]) -> List[str]:
        return [f'{entry["word"]}: {entry["text"]}' for entry in batch]


class PhiSentenceConnector(SentenceConnector):
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None

        self.generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def load_model(self):
        if self.pipe is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            model.eval()
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=self.device
            )

    def unload_model(self):
        if self.pipe is not None:
            del self.pipe
            torch.cuda.empty_cache()
            self.pipe = None

    def connect_word_text(self, word: str, text: str) -> str:
        return self.connect_batch([{'word': word, 'text': text}])[0]

    def connect_batch(self, batch: List[Dict]) -> List[str]:
        if not self.pipe:
            self.load_model()

        prompts = [self.get_prompt(entry['word'], entry['text']) for entry in batch]
        outputs = self.pipe(prompts, **self.generation_args)

        return [self.clean_output(entry, output[0]['generated_text'].strip()) for entry, output in zip(batch, outputs)]

    @staticmethod
    def get_prompt(word: str, text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are an assistant able to fuse two parts. You refrain from using word knowledge and adding not present facts. The first part is a word and the second part is a definition of the word."},
            {"role": "user", "content": "Fuse the following two parts split by MASK. The first part is a word and the second part is a definition of the word. If needed, use one of the following verbs: [is, represents, denotes, signifies, means, implies, symbolizes, describes, characterizes, embodies]. Only fuse them. Do not check factual correctness or correct spelling mistakes. Keep both parts in the same order and in the fused sentence.\nInput: Obama MASK president of Germany."},
            {"role": "assistant", "content": "Obama is the president of Germany."},
            {"role": "user", "content": "Fuse the following two parts split by MASK. The first part is a word and the second part is a definition of the word. If needed, use one of the following verbs: [is, represents, denotes, signifies, means, implies, symbolizes, describes, characterizes, embodies]. Only fuse them. Do not check factual correctness or correct spelling mistakes. Keep both parts in the same order and in the fused sentence.\nInput: Appel MASK A sweet, edible fruit produced by a tree, which is a member of the Rosaceae family."},
            {"role": "assistant", "content": "An Appel represents a sweet, edible fruit produced by a tree, which is a member of the Rosaceae family."},
            {"role": "user", "content": "Fuse the following two parts split by MASK. The first part is a word and the second part is a definition of the word. If needed, use one of the following verbs: [is, represents, denotes, signifies, means, implies, symbolizes, describes, characterizes, embodies]. Only fuse them. Do not check factual correctness or correct spelling mistakes. Keep both parts in the same order and in the fused sentence.\nInput: Starbright Foundation MASK Helps severely ill children."},
            {"role": "assistant", "content": "The Starbright Foundation helps severely ill children."},
            {"role": "user", "content": f"Fuse the following two parts split by MASK. The first part is a word and the second part is a definition of the word. If needed, use one of the following verbs: [is, represents, denotes, signifies, means, implies, symbolizes, describes, characterizes, embodies]. Only fuse them. Do not check factual correctness or correct spelling mistakes. Keep both parts in the same order and in the fused sentence.\nInput: {word} MASK {text}"},
        ]

    @staticmethod
    def clean_output(entry: Dict[str, str], output: str) -> str:
        if 'MASK' in output:
            output = output.replace('MASK', 'is')
        l_word = entry['word'].lower()
        l_output = output.lower()

        if not l_output.startswith(l_word) and not l_output.startswith((f'a {l_word}', f'an {l_word}', f'the {l_word}')):
            for pronoun in ('mine', 'your', 'his', 'her', 'its', 'our', 'their'):
                if l_output.startswith(pronoun):
                    output = f"A {output[len(pronoun) + 1:]}"
                    l_output = output.lower()
                    break
            if not l_output.startswith(l_word) and not l_output.startswith(f'a {l_word}') or entry['text'].lower() not in l_output:
                output = f"{entry['word']}: {entry['text']}"
        return output


if __name__ == "__main__":
    connector = PhiSentenceConnector()
    print(connector.connect_batch([
        {'word': 'Apple', 'text': 'red fruit.'},
        {'word': 'Banane', 'text': 'yellow fruit.'}
    ]))

    print(connector.connect_word_text(word='Apple', text='red fruit.'))
