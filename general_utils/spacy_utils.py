"""General utils using spacy for processing."""
import re
from typing import List, Tuple

import spacy
from spacy.tokens import Token

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
    """
    Returns a spacy Doc object for a given text in the specified language.

    :param txt: The input text.
    :param lang: Language of the input text ('en' for English, 'de' for German).
    :return: A spacy Doc object.
    """
    if lang == 'en':
        return nlp(txt)
    if lang == 'de':
        return german_nlp(txt)
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


def split_into_passage_sentences(text: str,
                                 sentence_limit: int = 3,
                                 lang: str = 'en') -> List[List[str]]:
    """
    Splits a text into passages, each containing a limited number of sentences.

    :param text: The input text.
    :param sentence_limit: Maximum number of sentences per passage.
    :param lang: Language of the text ('en' for English, 'de' for German).
    :return: A list of passages, each being a list of sentences.
    """
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


def get_words_before_root(sentence: str, lang: str = 'en') -> str:
    """
    Extracts all words before the root word in a sentence.

    :param sentence: The input sentence.
    :param lang: Language of the sentence ('en' for English, 'de' for German).
    :return: A string containing all words before the root.
    """
    doc = get_doc(sentence, lang)

    tokens = []
    for token in doc:
        if token.dep_ == "ROOT":
            break
        tokens.append(token.norm_)
    return process_sentence(' '.join(tokens)).strip()


def get_words_after_root(sentence: str, lang: str = 'en') -> str:
    """
    Extracts all words after the root word in a sentence.

    :param sentence: The input sentence.
    :param lang: Language of the sentence ('en' for English, 'de' for German).
    :return: A string containing all words after the root.
    """
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


def tok_format(tok: Token) -> str:
    """
    Returns the original text of a token.

    :param tok: A spacy token object.
    :return: The original text of the token.
    """
    return f"{tok.orth_}"


def to_nltk_tree(node: Token) -> dict | str:
    """
    Converts a spacy dependency tree into an NLTK-style tree format.

    :param node: A spacy token node.
    :return: An NLTK-style dependency tree.
    """
    if node.n_lefts + node.n_rights > 0:
        return {tok_format(node): [to_nltk_tree(child) for child in node.children]}
    return tok_format(node)


def remove_starting_article(txt: str, lang: str = 'en') -> str:
    """
    Removes the starting article (determiner) from a text.

    :param txt: The input text.
    :param lang: Language of the text ('en' for English, 'de' for German).
    :return: The text without the starting article.
    """
    doc = get_doc(txt, lang)

    first_token = doc[0]
    if first_token.pos_ == 'DET':
        txt = txt[len(first_token) + 1:]
        return txt
    return txt


def is_single_word(txt: str, lang: str = 'en') -> bool:
    """
    Determines if the text is a single word or a named entity.

    :param txt: The input text.
    :param lang: Language of the text ('en' for English, 'de' for German).
    :return: True if the text is a single word or a named entity, False otherwise.
    """
    doc = get_doc(txt, lang)

    if len(doc) == 1:
        return True

    # Check if the entire phrase is a named entity
    if any(ent.text == txt for ent in doc.ents):
        return True

    return False


def create_german_fact(question_sent: str, answer_sent: str) -> Tuple[str, str] | None:
    """
    Creates a fact sentence from a German question and answer.

    :param question_sent: The input question sentence.
    :param answer_sent: The input answer sentence.
    :return: A tuple containing the fact sentence and the entity, or None if no fact is created.
    """
    question_conversion = {"Was ist": "{} ist {}.",
                           "Was bezeichnet man als": "Als {} bezeichnet man {}.",
                           "Was bezeichnet": "{} bezeichnet {}.",
                           "Was bedeutet": "{} bedeuet {}.",
                           "Was macht": "{} macht {}.",
                           "Was kennzeichnet": "{} kennzeichnet {}."
                           }

    for key, value in question_conversion.items():
        if question_sent.startswith(key):
            entity = question_sent[len(key) + 1: -1]
            entity = remove_starting_article(entity, lang='de')  # remove leading article
            if not is_single_word(entity, lang='de'):  # we only want single words
                continue

            fact_sent = value.format(entity, answer_sent).strip()
            fact_sent = fact_sent[0].upper() + fact_sent[1:]
            return fact_sent, entity
    return None


def create_english_fact(question_sent: str, answer_sent: str) -> Tuple[str, str] | None:
    """
    Creates a fact sentence from an English question and answer.

    :param question_sent: The input question sentence.
    :param answer_sent: The input answer sentence.
    :return: A tuple containing the fact sentence and the entity, or None if no fact is created.
    """
    question_conversion = {
        r"What is (.*)": "{} is {}.",
        r"What is referred to as (.*)": "{} is referred to as {}.",
        r"What refers to (.*)": "{} refers to {}.",
        r"What means (.*)": "{} means {}.",
        r"What does (.*) mean\?": "{} means {}.",
        r"What does (.*)": "{} does {}.",
        r"What characterizes (.*)": "{} characterizes {}."
    }

    for pattern, value in question_conversion.items():
        match = re.search(pattern, question_sent)

        if match:
            entity = match.group(1)
            entity = remove_starting_article(entity, lang='en')  # remove leading article
            if not is_single_word(entity, lang='en'):  # we only want single words
                continue

            fact_sent = value.format(entity, answer_sent).strip()
            fact_sent = fact_sent[0].upper() + fact_sent[1:]
            return fact_sent, entity
    return None


def is_german_def_question(question_sent: str) -> bool:
    """
    Checks if a German question asks for a definition.

    :param question_sent: The input question sentence.
    :return: True if the question asks for a definition, False otherwise.
    """
    question_conversion = {"Was ist": "{} ist {}.",
                           "Was bezeichnet man als": "Als {} bezeichnet man {}.",
                           "Was bezeichnet": "{} bezeichnet {}.",
                           "Was bedeutet": "{} bedeuet {}.",
                           "Was macht": "{} macht {}.",
                           "Was kennzeichnet": "{} kennzeichnet {}."
                           }

    if question_sent.startswith(tuple(question_conversion.keys())):
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


def get_main_entity(txt: str, lang: str = 'en'):
    doc = get_doc(txt, lang)

    if doc.ents:
        return doc.ents[0].text

    for token in doc:
        if token.dep_ in {"nsubj", "ROOT", "pobj", "dobj"} and token.pos_ == "NOUN":
            return token.text
    return None
