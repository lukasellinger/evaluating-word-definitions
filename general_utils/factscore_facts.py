"""Module for FactScore Facts"""
import json
import numpy as np
import re
import string
import spacy
from rank_bm25 import BM25Okapi
import os


class FactScoreFactGenerator(object):
    def __init__(self, demon_dir, is_bio=True):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = is_bio
        self.demon_path = os.path.join(demon_dir,
                                       "demons.json" if self.is_bio else "demons_complex.json")

        # get the demos
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_prompt_for_sentence(self, sentence):
        """Get the prompt for fact splitting of the sentence."""

        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
        prompt = ""

        for i in range(n):
            prompt += (f"Please breakdown the following sentence into independent facts: "
                       f"{list(demons.keys())[i]}\n")
            for fact in demons[list(demons.keys())[i]]:
                prompt += f"- {fact}\n"
            prompt += "\n"

        for match in top_machings:
            prompt += f"Please breakdown the following sentence into independent facts: {match}\n"
            for fact in demons[match]:
                prompt += f"- {fact}\n"
            prompt = prompt + "\n"
        prompt += f"Please breakdown the following sentence into independent facts: {sentence}\n"
        return prompt

    def get_prompt_for_sentence2(self, sentence):
        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        prompts = []
        prompts.append({"role": "user",
                        'content': "Please breakdown the following sentence into independent "
                                   "facts. Do not be too finegrained. Refrain from introducing new "
                                   "facts. Refrain from using world knowledge:\n Empire State "
                                   "Building: personal essay about Woody Allen\n"})
        prompts.append({"role": "assistant",
                        'content': "1. Empire State Building is a personal essay.\n"
                                   "2. Empire State Building is about Woody Allen."})
        prompts.append({"role": "user",
                        'content': "Please breakdown the following sentence into independent "
                                   "facts. Do not be too finegrained. Refrain from introducing "
                                   "new facts. Refrain from using world knowledge:\n "
                                   "Marilyn Monroe: part of the war effort\n"})
        prompts.append({"role": "assistant",
                        'content': "1. Marilyn Monroe was a part of the war effort."})
        prompts.append({"role": "user",
                        'content': "Please breakdown the following sentence into independent "
                                   "facts. Do not be too finegrained. Refrain from introducing "
                                   "new facts. Refrain from using world knowledge:\n "
                                   "Asthma: audio form of marketing communication.\n"})
        prompts.append({"role": "assistant",
                        'content': "1. Asthma is an audio form.\n"
                                   "2. Asthma is an form of marketing communication."})
        prompts.append({"role": "user",
                        'content': "Please breakdown the following sentence into independent "
                                   "facts. Do not be too finegrained. Refrain from introducing "
                                   "new facts. Refrain from using "
                                   f"world knowledge:\n {sentence}\n"})
        return prompts

    def get_facts_from_response(self, sentence, model_response: str):
        facts = text_to_sentences(model_response)

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        if self.is_bio:
            facts = postprocess_atomic_facts(sentence, facts, self.nlp)

        return facts


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# transform InstructGPT output into sentences
def text_to_sentences(text):
    text = text.replace('\\n', '\n')
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in
                 sentences]
    if len(sentences) > 0:
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.'
    else:
        sentences = []
    return sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]


def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False


def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True


def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set(
        [value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)

    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)

    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities


def postprocess_atomic_facts(sent, atomic_facts, nlp):
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.",
             " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    new_atomic_facts = []

    entities = detect_entities(sent, nlp)
    covered_entities = set()

    new_facts = []
    for i, fact in enumerate(atomic_facts):
        if any([fact.endswith(verb) for verb in verbs]) and not any(
                [fact.endswith(verb) for verb in permitted_verbs]):
            if any([fact[:-1] in other_fact for j, other_fact in enumerate(atomic_facts) if j != i]):
                continue
        sent_entities = detect_entities(fact, nlp)
        covered_entities |= set([e for e in sent_entities if e in entities])
        new_entities = sent_entities - entities
        if len(new_entities) > 0:
            do_pass = False
            for new_ent in new_entities:
                pre_ent = None
                for ent in entities:
                    if ent.startswith(new_ent):
                        pre_ent = ent
                        break
                if pre_ent is None:
                    do_pass = True
                    break
                fact = fact.replace(new_ent, pre_ent)
                covered_entities.add(pre_ent)
            if do_pass:
                continue
        if fact in new_facts:
            continue
        new_facts.append(fact)
    try:
        assert entities == covered_entities
    except Exception:
        new_facts = atomic_facts  # there is a bug in spacy entity linker, so just go with the previous facts

    new_atomic_facts.append(new_facts)

    return new_atomic_facts
