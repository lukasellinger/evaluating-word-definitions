"""Module for claim splitters."""
import re
from abc import ABC, abstractmethod
from typing import List

import requests
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import HF_READ_TOKENS, PROJECT_DIR
from fetchers.openai import OpenAiFetcher
from general_utils.factscore_facts import FactScoreFactGenerator
from general_utils.spacy_utils import split_into_sentences
from general_utils.utils import sentence_simplification


class ClaimSplitter(ABC):
    """
    Abstract base class for claim splitters.

    Defines the interface for claim splitting, providing methods for both
    single text and batch processing of atomic claims.
    """

    def __call__(self, batch):
        return self.get_atomic_claims_batch(batch)

    @abstractmethod
    def get_atomic_claims(self, text: str) -> dict:
        """
        Obtain atomic claims from a single text.

        :param text: The input text to split into atomic claims.
        :return: A list of atomic claims.
        """

    @abstractmethod
    def get_atomic_claims_batch(self, texts: List[str]) -> List[dict]:
        """
        Obtain atomic claims from a batch of texts.

        :param texts: List of texts to split into atomic claims.
        :return: A list of lists, where each list contains atomic claims for a given text.
        """


class DisSimSplitter(ClaimSplitter):
    """DisSim Claim Splitter https://github.com/Lambda-3/DiscourseSimplification"""
    def get_atomic_claims(self, text: str) -> dict:
        output = sentence_simplification([text])
        return output[0]

    def get_atomic_claims_batch(self, texts: List[str]) -> List[dict]:
        return sentence_simplification(texts)


class MixtralSplitter(ClaimSplitter):
    """unikei/t5-base-split-and-rephrase Claim Splitter"""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    HF_TOKEN = HF_READ_TOKENS[0]

    def __init__(self, api_url=None, hf_token=None):
        if api_url:
            self.API_URL = api_url
        if hf_token:
            self.HF_TOKEN = hf_token

    def get_atomic_claims_batch(self, texts: List[str]) -> List[dict]:
        return [self.get_atomic_claims(text) for text in texts]

    def get_atomic_claims(self, text: str) -> dict:
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
        return {'error': 'No generated text.'}

    def query(self, payload, token):
        headers = {"Authorization": "Bearer {token}"}
        headers['Authorization'] = headers['Authorization'].format(token=token)
        response = requests.post(self.API_URL, headers=headers, json=payload, timeout=10)
        return response.json()

    @staticmethod
    def get_prompt(txt: str) -> str:
        """Gets the prompt for the text you want to split."""
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
    """unikei/t5-base-split-and-rephrase Claim Splitter"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "unikei/t5-base-split-and-rephrase"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

    def unload_model(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def get_atomic_claims(self, text: str) -> dict:
        return self.get_atomic_claims_batch([text])[0]

    def get_atomic_claims_batch(self, texts: List[str]) -> List[dict]:
        if not self.model:
            self.load_model()

        complex_tokenized = self.tokenizer(texts,
                                           padding="longest",
                                           truncation=True,
                                           max_length=256,
                                           return_tensors='pt').to(self.device)

        simple_tokenized = self.model.generate(complex_tokenized['input_ids'],
                                               attention_mask=complex_tokenized['attention_mask'],
                                               max_length=256,
                                               num_beams=5)
        simple_texts = self.tokenizer.batch_decode(simple_tokenized, skip_special_tokens=True)
        return [{'text': text, 'splits': list(set(split_into_sentences(simple)))} for text, simple in zip(texts, simple_texts)]


class FactscoreSplitter(ClaimSplitter):
    """FActScore Splitter from https://aclanthology.org/2023.emnlp-main.741/"""
    def __init__(self, api_key=None):
        self.openai_fetcher = OpenAiFetcher(api_key=api_key)
        self.prompt_generator = FactScoreFactGenerator(PROJECT_DIR.joinpath('factscore/demos'),
                                                       is_bio=False)

    def get_atomic_claims(self, text: str) -> dict:
        return self.get_atomic_claims_batch([text])[0]

    def get_atomic_claims_batch(self, texts: List[str]) -> List[dict]:
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
