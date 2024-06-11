"""General utils using spacy for processing."""
from typing import List

import spacy

nlp = spacy.load("en_core_web_lg")
german_nlp = spacy.load("de_core_news_lg")


def get_doc(txt: str, lang: str = 'en'):
    if lang == 'en':
        return nlp(txt)
    elif lang == 'de':
        return german_nlp(txt)
    else:
        raise ValueError(f'Language {lang} not supported.')


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

    root_token = 'ROOT'
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = ' ' + str(token) + ' '
            break

    return sentence.split(root_token)[0].strip()

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
