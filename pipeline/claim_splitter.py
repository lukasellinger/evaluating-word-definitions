import re
from abc import ABC, abstractmethod
from typing import List, Dict

import requests
from transformers import T5Tokenizer, T5ForConditionalGeneration

from config import HF_READ_TOKENS, PROJECT_DIR
from fetchers.openai import OpenAiFetcher
from general_utils.atomic_facts import FactScoreFactGenerator
from general_utils.spacy_utils import split_into_sentences
from general_utils.utils import sentence_simplification


class ClaimSplitter(ABC):

    def __call__(self, batch):
        return self.get_atomic_claims_batch(batch)

    @abstractmethod
    def get_atomic_claims(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def get_atomic_claims_batch(self, texts: List[str]) -> List[List[str]]:
        pass


class DisSimSplitter(ClaimSplitter):
    def get_atomic_claims(self, text: str) -> Dict:
        output = sentence_simplification([text])
        return output[0]

    def get_atomic_claims_batch(self, texts: List[str]) -> List[Dict]:
        return sentence_simplification(texts)


class MixtralSplitter(ClaimSplitter):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    HF_TOKEN = HF_READ_TOKENS[0]

    def __init__(self, api_url=None, hf_token=None):
        if api_url:
            self.API_URL = api_url
        if hf_token:
            self.HF_TOKEN = hf_token

    def get_atomic_claims_batch(self, texts: List[str]) -> List[Dict]:
        return [self.get_atomic_claims(text) for text in texts]

    def get_atomic_claims(self, text: str) -> Dict:
        output = self.query({'inputs': self.get_prompt(text),
                             'parameters': {'temperature': 0.01, 'return_full_text': False}},
                            token=self.HF_TOKEN)
        if isinstance(output, dict):
            return {'error': output.get('error')}

        answer = output[0].get('generated_text')
        if answer:
            facts = []
            explanation = ""
            for line in answer.split('\n'):
                line = line.strip()
                fact_match = re.match(r'^\d+\.(.*)', line)
                explanation_match = re.match(r'^Explanation: (.*)', line)

                if fact_match:
                    fact = fact_match.group(1).strip()
                    facts.append(fact)
                elif explanation_match:
                    explanation = explanation_match.group(1).strip()
            return {'text': text, 'splits': facts, 'explanation': explanation}

    def query(self, payload, token):
        headers = {"Authorization": "Bearer {token}"}
        headers['Authorization'] = headers['Authorization'].format(token=token)
        response = requests.post(self.API_URL, headers=headers, json=payload)
        return response.json()

    @staticmethod
    def get_prompt(txt: str):
        prompt = """<s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: Mammals are vertebrates and encompass cars. [/INST]
        1. Mammals are vertebrates.
        2. Mammals encompass cars.

        Explanation: Split based on the two distinct characteristics described about mammals - being vertebrates and supposedly encompassing cars.
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: Tottenham Hotspur F.C. is a basketball team and often a elephant. [/INST]
        1. Tottenham Hotspur F.C. is a basketball team.
        2. Tottenham Hotspur F.C. is often a elephant

        Explanation: Two separate attributes are assigned to Tottenham Hotspur F.C., one regarding the type of team it is and another portraying it as an elephant at times.
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: Marilyn Monroe was a part of the war effort. [/INST]
        1. Marilyn Monroe was a part of the war effort.

        Explanation: The statement links Marilyn Monroe with participation in a war effort. As there's no other claim to separate, the entire statement stands as a single independent fact.    
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: LinkedIn is available in zero languages as of 2013. [/INST]
        1. LinkedIn is available in zero languages.
        2. This availability was true as of 2013.

        Explanation: The statement links two distinct, independently verifiable facts about LinkedIn: one referring to the number of languages it supports, and one indicating a timeline.    
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: Haifa was unknown as a dye-making center. [/INST]
        1. Haifa was unknown.
        2. Haifa was a dye-making center.

        Explanation: Statement claims two separate facts: one about Haifa's recognition or lack thereof, and one claiming it as a dye-making center.
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: This Is Us has received nominations for Best Television Series Drama. [/INST]
        1. This Is Us has received nominations.
        2. Nominations for Best Television Series Drama.

        Explanation: Statement describes both the event (receiving nominations) and the specific category for that event which can be separately verified.
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: Match Point was a personal essay about Woody Allen. [/INST]
        1. Match Point was a personal essay.
        2. Match Point is about Woody Allen.

        Explanation: The statement asserts the nature of Match Point as an essay, and additionally clarifies the subject matter of this essay, namely Woody Allen.
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: Advertising is an audio form of marketing communication. [/INST]
        1. Advertising is an audio form.
        2. Advertising is an form of marketing communication.

        Explanation: The statement outlines the method of advertising (via audio) while also noting the broader classification of advertising as a marketing communication tool. These two aspects can be verified independently.
        </s> [INST] Please deconstruct the following statement into its main distinct autonomous facts. Refrain from using any external resources. You are not responsible for evaluating the truthfulness or correctness of these facts; your task is only to identify them. Do not be too finegrained: {sentence} [/INST]
        """
        return prompt.format(sentence=txt)


class T5SplitRephraseSplitter(ClaimSplitter):

    def __init__(self):
        checkpoint = "unikei/t5-base-split-and-rephrase"
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    def get_atomic_claims(self, text: str) -> Dict:
        return self.get_atomic_claims_batch([text])[0]

    def get_atomic_claims_batch(self, texts: List[str]) -> List[Dict]:
        complex_tokenized = self.tokenizer(texts,
                                           padding="longest",
                                           truncation=True,
                                           max_length=256,
                                           return_tensors='pt')

        simple_tokenized = self.model.generate(complex_tokenized['input_ids'],
                                               attention_mask=complex_tokenized['attention_mask'],
                                               max_length=256,
                                               num_beams=5)
        simple_texts = self.tokenizer.batch_decode(simple_tokenized, skip_special_tokens=True)
        return [{'text': text, 'splits': list(set(split_into_sentences(simple)))} for text, simple in zip(texts, simple_texts)]


class FactscoreSplitter(ClaimSplitter):

    def __init__(self, api_key=None):
        self.openai_fetcher = OpenAiFetcher(api_key=api_key)
        self.prompt_generator = FactScoreFactGenerator(PROJECT_DIR.joinpath('factscore/demos'),
                                                       is_bio=False)

    def get_atomic_claims(self, text: str) -> Dict:
        return self.get_atomic_claims_batch([text])[0]

    def get_atomic_claims_batch(self, texts: List[str]) -> List[Dict]:
        model = "gpt-3.5-turbo-instruct"
        temperature = 0.7
        max_tokens = 512

        facts_batch = []
        for text in texts:
            prompt = self.prompt_generator.get_prompt_for_sentence(text)
            response = self.openai_fetcher.client.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt=prompt
            )
            generated_answer = response.choices[0].text
            facts = self.prompt_generator.get_facts_from_response(text, generated_answer)
            facts_batch.append({'text': text, 'splits': facts})
        return facts_batch


if __name__ == "__main__":
    splitter = FactscoreSplitter()
    print(splitter.get_atomic_claims_batch([
        "Alice likes soccer and Bob likes tennis."
    ]))
