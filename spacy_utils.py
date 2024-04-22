"""General utils using spacy for processing."""
from typing import List

import spacy

nlp = spacy.load("en_core_web_lg")
german_nlp = spacy.load("de_core_news_lg")


def recognize_definition(sentence: str, simple=False) -> bool:
    """Check whether sentence is a definition."""
    definitions_keywords = [
        "be",
        "represent",
        "denote",
        "refer",
        "signify",
        "constitute",
        "mean",
        "stand",
        "imply",
        "equal",
        "symbolize",
        "describe",
        "manifest",
        "correspond",
        "characterize",
        "epitomize",
        "exemplify",
        "embody",
        "portray"
    ]

    if simple:
        return sentence and sentence.split(' ')[1] in definitions_keywords

    doc = nlp(sentence)

    first_token = doc[0]
    if first_token.dep_ == 'compound':
        first_token = first_token.head

    if first_token.dep_ == "nsubj":
        subject_head = first_token.head

        if subject_head.dep_ == "ROOT":
            # Check if the head of the subject is a verb from the given list
            if subject_head.lemma_.lower() in definitions_keywords:
                first_token.children
                return True
    return False


def words_to_lemmas(words: List[str]) -> List[str]:
    """Convert all words to their lemma."""
    lemmatizer = nlp.get_pipe('lemmatizer')

    keyword_lemmas = []
    for keyword in words:
        word = nlp(keyword)
        word[0].pos_ = 'VERB'

        lemma = []
        for token in word:
            token.pos_ = 'VERB'
            lemma.append(''.join(lemmatizer.lemmatize(token)))
        keyword_lemmas.append(''.join(lemma))
    return keyword_lemmas


def check_person(sentence: str) -> bool:
    """Check if first token belongs to a person."""
    return get_ent_type(sentence) == 'PERSON'


def get_ent_type(sentence: str) -> str:
    """Get named entity type of first token."""
    doc = nlp(sentence)

    first_token = doc[0]
    if first_token.dep_ == 'compound':
        first_token = first_token.head

    return first_token.ent_type_
