"""General utils using spacy for processing."""
import re
from typing import List

import spacy

from general_utils.utils import process_sentence

nlp = spacy.load("en_core_web_lg")
german_nlp = spacy.load("de_core_news_lg")

DEFINITION_KEYWORDS = [
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


def get_doc(txt: str, lang: str = 'en'):
    if lang == 'en':
        return nlp(txt)
    elif lang == 'de':
        return german_nlp(txt)
    else:
        raise ValueError(f'Language {lang} not supported.')


def recognize_definition(sentence: str, simple=False) -> bool:
    """Check whether sentence is a definition."""
    if simple:
        return sentence and sentence.split(' ')[1] in DEFINITION_KEYWORDS

    doc = nlp(sentence)

    first_token = doc[0]
    if first_token.dep_ == 'compound':
        first_token = first_token.head

    if first_token.dep_ == "nsubj":
        subject_head = first_token.head

        if subject_head.dep_ == "ROOT":
            # Check if the head of the subject is a verb from the given list
            if subject_head.lemma_.lower() in DEFINITION_KEYWORDS:
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


def check_person(sentence: str, lang: str = 'en') -> bool:
    """Check if first token belongs to a person."""
    return get_ent_type(sentence, lang) == 'PERSON'


def get_ent_type(sentence: str, lang: str = 'en') -> str:
    """Get named entity type of first token."""
    doc = get_doc(sentence, lang)

    first_token = doc[0]
    if first_token.dep_ == 'compound':
        first_token = first_token.head

    return first_token.ent_type_


def split_into_sentences(txt: str, lang: str = 'en') -> List[str]:
    """Split a text into sentences."""
    doc = get_doc(txt, lang)
    return [sent.text.strip() for sent in doc.sents]


def split_into_passage_sentences(text: str, sentence_limit: int = 3, lang: str = 'en'):
    sentences = split_into_sentences(text, lang)
    passages = []
    for i in range(0, len(sentences), sentence_limit):
        passages.append(sentences[i:i + sentence_limit])

    return passages


def get_first_entity(txt: str, lang: str = 'en') -> str:
    """Takes a text and return the first entity in the text."""
    doc = get_doc(txt, lang)
    first_entity = doc.ents[0].text if doc.ents else None
    return first_entity


def get_first_compound_or_word(txt: str, lang: str = 'en') -> str:
    """Take a text and return the first compound word or the first standalone word in the text.
    """
    doc = get_doc(txt, lang)

    compound = []
    for token in doc:
        if token.dep_ == "compound":
            compound.append(str(token))
        else:
            compound.append(str(token))
            break
    return " ".join(compound)


def get_words_before_root(sentence: str, lang='en') -> str:
    """Get all words before the root of the sentence"""
    doc = get_doc(sentence, lang)

    tokens = []
    for token in doc:
        if token.dep_ == "ROOT":
            break
        else:
            tokens.append(token.norm_)
    return process_sentence(' '.join(tokens)).strip()


def get_words_after_root(sentence: str, lang='en') -> str:
    """Get all words after the root of the sentence"""
    doc = get_doc(sentence, lang)

    tokens = []
    root_occured = False  # sometimes spacy sentences has multiple roots. we take the first one
    for token in doc:
        if token.dep_ == "ROOT":
            root_occured = True
            continue
        if root_occured:
            tokens.append(token.norm_)

    return process_sentence(' '.join(tokens)).strip()


def tok_format(tok):
    """Get original verbatim text of tok."""
    return f"{tok.orth_}"


def to_nltk_tree(node):
    """Get the dependency parse tree."""
    if node.n_lefts + node.n_rights > 0:
        return {tok_format(node): [to_nltk_tree(child) for child in node.children]}
    else:
        return tok_format(node)


def remove_starting_article(txt: str, lang='en'):
    doc = get_doc(txt, lang)

    first_token = doc[0]
    if first_token.pos_ == 'DET':
        txt = txt[len(first_token) + 1:]
        return txt
    return txt


def is_single_word(txt: str, lang='en'):
    doc = get_doc(txt, lang)

    if len(doc) == 1:
        return True

    # Check if the entire phrase is a named entity
    if any(ent.text == txt for ent in doc.ents):
        return True

    return False


def create_german_fact(question_sent, answer_sent):
    """Create a fact sentence out of the question and answer."""
    QUESTION_CONVERSION = {"Was ist": "{} ist {}.",
                           "Was bezeichnet man als": "Als {} bezeichnet man {}.",
                           "Was bezeichnet": "{} bezeichnet {}.",
                           "Was bedeutet": "{} bedeuet {}.",
                           "Was macht": "{} macht {}.",
                           "Was kennzeichnet": "{} kennzeichnet {}."
                           }

    for key, value in QUESTION_CONVERSION.items():
        if question_sent.startswith(key):
            entity = question_sent[len(key) + 1: -1]
            entity = remove_starting_article(entity, lang='de')  # remove leading article
            if not is_single_word(entity, lang='de'):  # we only want single words
                continue

            fact_sent = value.format(entity, answer_sent).strip()
            fact_sent = fact_sent[0].upper() + fact_sent[1:]
            return fact_sent, entity


def create_english_fact(question_sent, answer_sent):
    """Checks whether the question asks for a definition of a word and creates a fact out of it."""
    QUESTION_CONVERSION = {
        r"What is (.*)": "{} is {}.",
        r"What is referred to as (.*)": "{} is referred to as {}.",
        r"What refers to (.*)": "{} refers to {}.",
        r"What means (.*)": "{} means {}.",
        r"What does (.*) mean\?": "{} means {}.",
        r"What does (.*)": "{} does {}.",
        r"What characterizes (.*)": "{} characterizes {}."
    }

    for pattern, value in QUESTION_CONVERSION.items():
        match = re.search(pattern, question_sent)

        if match:
            entity = match.group(1)
            entity = remove_starting_article(entity, lang='en')  # remove leading article
            if not is_single_word(entity, lang='en'):  # we only want single words
                continue

            fact_sent = value.format(entity, answer_sent).strip()
            fact_sent = fact_sent[0].upper() + fact_sent[1:]
            return fact_sent, entity


def is_german_def_question(question_sent):
    """Checks whether the entry has a question which asks for a definition of a word."""
    QUESTION_CONVERSION = {"Was ist": "{} ist {}.",
                           "Was bezeichnet man als": "Als {} bezeichnet man {}.",
                           "Was bezeichnet": "{} bezeichnet {}.",
                           "Was bedeutet": "{} bedeuet {}.",
                           "Was macht": "{} macht {}.",
                           "Was kennzeichnet": "{} kennzeichnet {}."
                           }

    if question_sent.startswith(tuple(QUESTION_CONVERSION.keys())):
        if question_sent.startswith('Was ist'):
            question_tokens = german_nlp(question_sent)

            word = question_tokens[2]
            if word.pos_ == 'DET':
                word = word.head
            if word.dep_ == 'compound':
                word = word.head
            for child in word.children:
                if child.pos_ == 'ADJ':
                    return False

            if question_sent.endswith(f'{word}?'):
                return True
        else:
            return True
    return False
