import re
from typing import Dict

import requests

from config import HF_READ_TOKENS


class FactExtractor:
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
    HF_TOKEN = HF_READ_TOKENS[0]

    def __init__(self, api_url=None, hf_token=None):
        if api_url:
            self.API_URL = api_url
        if hf_token:
            self.HF_TOKEN = hf_token

    def get_atomic_facts(self, txt: str) -> Dict:
        output = self.query({'inputs': self.get_prompt(txt),
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
            return {'facts': facts, 'explanation': explanation}

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


if __name__ == "__main__":
    extractor = FactExtractor()
    print(extractor.get_atomic_facts('Nudossi is a brand of chocolate cream from the GDR. The answer would be: "Nudossi is a brand of chocolate cream" or simply "DDR chocolate cream".'))