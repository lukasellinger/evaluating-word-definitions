"""General utils for processing."""
import itertools
import re
import string
import subprocess
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score, f1_score

from config import PROJECT_DIR
from general_utils.reader import LineReader, JSONReader


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")

    raise ValueError(f"Unsupported string type: {type(text)}")


def rank_docs(query: str, docs: List[str], k=5, get_indices=True) -> List[str] | List[int]:
    """
    Get the top k most similar documents according to the query using the BM25 algorithms.
    :param query: query sentence.
    :param docs: documents to rank.
    :param k: amount of documents to return
    :param get_indices: If True, returns the indices, else the text.
    :return: List of most similar documents.
    """
    def preprocess(txt: str):
        return txt.lower()

    query = preprocess(query)
    docs = [preprocess(doc) for doc in docs]
    tokenized_corpus = [doc.split(" ") for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    if get_indices:
        scores = np.array(bm25.get_scores(query.split(" ")))
        return np.flip(np.argsort(scores)[-k:]).tolist()
    return bm25.get_top_n(query.split(" "), docs, k)


def convert_document_id_to_word(document_id: str) -> str:
    """Converts a document_id to a word we would search for."""
    word = document_id.split('-LRB-')[0]
    return word.replace('_', ' ')


def calc_bin_stats(gt_labels: List, pr_labels: List, values: List) -> Dict:
    """
    Calculate the stats for each bin. Bins a separated using sturgess rule.
    :param gt_labels: ground truth labels.
    :param pr_labels: predicted labels.
    :param values: value to an according ground truth / predicted pair.
    :return: A dictionary of bins and their stats.
    """
    values = np.array(values)
    gt_labels = np.array(gt_labels)
    pr_labels = np.array(pr_labels)
    bin_stats = {}

    m = round(1 + np.log2(len(values)))  # sturgess rule
    max_l = np.max(values)
    min_l = np.min(values)
    bin_size = (max_l - min_l) / m
    for k in range(m):
        bin_lower = min_l + k * bin_size
        bin_upper = bin_lower + bin_size
        bin_mask = (bin_lower <= values) & (values < bin_upper)
        bin_pr_labels = pr_labels[bin_mask]
        bin_gt_labels = gt_labels[bin_mask]

        if len(bin_pr_labels) > 0:
            bin_stats[bin_upper] = {'acc': accuracy_score(bin_gt_labels, bin_pr_labels),
                                    'f1_weighted': f1_score(bin_gt_labels, bin_pr_labels,
                                                            average='weighted'),
                                    'f1_macro': f1_score(bin_gt_labels, bin_pr_labels,
                                                         average='macro')}
    return bin_stats


def plot_graph(keys, values, x_label='', y_label='', title=''):
    """Plots a graph. Keys are associated with x-axis, values with y-axis."""
    plt.plot(keys, values, marker='o', linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(keys, rotation=45)
    plt.show()


def build_fever_instance(label, evidence, evidence_doc, predicted_label, predicted_evidence):
    """Build instance to conform to fever scorer."""
    evidence = [
        [
            [None, None, evidence_doc, int(line)]
            for line in group.split(',')
        ]
        for group in evidence
    ]
    predicted_evidence = [[page, int(line)] for page, line in predicted_evidence]

    instance = {'label': label,
                'predicted_label': predicted_label.name,
                'predicted_evidence': predicted_evidence,
                'evidence': evidence}
    return instance


def title_to_db_page(txt: str, parentheses=True) -> str:
    """Converts a title to how it is stored in the db."""
    txt = txt.replace(' ', '_')
    if parentheses:
        txt = txt.replace('-', '--')
        txt = txt.replace('(', '-LRB-')
        txt = txt.replace(')', '-RRB-')
        txt = txt.replace(':', '-COLON-')

    return txt


def remove_non_alphabetic_start_end(text):
    text = text.strip()
    text = re.sub(r'[^a-zA-Z]+$', '', text)
    text = re.sub(r'^[^a-zA-Z]+', '', text)
    return text.strip()


def pretty_string_list(lst: List) -> str:
    output = ""
    for item in lst:
        output += str(item) + '\n'
    return output


def generate_case_combinations(txt: str) -> List[str]:
    words = txt.split()
    combinations = []

    # Generate all combinations of upper and lower case for the first letter of each word
    for case_pattern in itertools.product(*[(word[0].lower(), word[0].upper()) for word in words]):
        # Reconstruct the sentence with the current case pattern
        combination = " ".join(
            pattern + word[1:] for pattern, word in zip(case_pattern, words)
        )
        combinations.append(combination)

    return combinations


def sentence_simplification(sentences: List[str]):
    discourse_simplification = PROJECT_DIR.joinpath('../DiscourseSimplification')
    LineReader().write(discourse_simplification.joinpath('input.txt'), sentences, mode='w')
    command = ["mvn", "-f", discourse_simplification.joinpath("pom.xml"), "clean", "compile", "exec:java"]
    subprocess.run(command, text=True, cwd=discourse_simplification)
    outputs = JSONReader().read(discourse_simplification.joinpath('output.json')).get('sentences')
    outputs = [{'sentence': entry.get('originalSentence'),
                'splits': [split.get('text') for split in entry.get('elementMap').values()]}
               for entry in outputs]
    return outputs


def find_substring_in_list(lst, substring):
    for index, string in enumerate(lst):
        if substring in string:
            return index
    return -1


def process_sentence_wiki(sentence):
    """Converts characters to their original representation in a sentence."""
    sentence = convert_to_unicode(sentence)
    sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
    sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
    sentence = re.sub("-LRB-", "(", sentence)
    sentence = re.sub("-RRB-", ")", sentence)
    sentence = re.sub("-COLON-", ":", sentence)
    sentence = re.sub("_", " ", sentence)
    sentence = re.sub(r"\( *\,? *\)", "", sentence)
    sentence = re.sub(r"\( *[;,]", "(", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


def process_sentence(sentence):
    """Converts characters to their original representation in a sentence."""
    sentence = convert_to_unicode(sentence)
    sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
    sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
    sentence = re.sub("-LRB-\s*", "(", sentence)
    sentence = re.sub("\s*-RRB-", ")", sentence)
    sentence = re.sub(r"\(\s*", "(", sentence)
    sentence = re.sub(r"\s*\)", ")", sentence)
    sentence = re.sub("\s*-COLON-\s*", ":", sentence)
    sentence = re.sub("_", " ", sentence)
    sentence = re.sub(r"\( *\,? *\)", "", sentence)
    sentence = re.sub(r"\( *[;,]", "(", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub(r"``\s*", '"', sentence)
    sentence = re.sub(r"\s*''", '"', sentence)
    sentence = re.sub(r" \.", '.', sentence)
    sentence = re.sub(r" ,", ',', sentence)
    sentence = re.sub(r" ;", ';', sentence)
    sentence = re.sub(r" :", ':', sentence)
    return sentence


def split_into_passages(text, passage_length=256):
    words = text.split()
    passages = [' '.join(words[i:i + passage_length]) for i in range(0, len(words), passage_length)]
    return passages


def remove_duplicate_values(d: Dict):
    """
    Remove duplicate values from the dictionary, keeping only the first occurrence of each value.
    """
    seen_values = set()
    unique_dict = {}

    for key, value in d.items():
        if isinstance(value, list):
            value_tuple = tuple(value)  # Convert list to tuple
        else:
            value_tuple = value  # Use the value directly if it's not a list

        if value_tuple not in seen_values:
            unique_dict[key] = value
            seen_values.add(value_tuple)
    return unique_dict


def parse_model_answer(generated_answer: str):
    # when logits are unavailable
    generated_answer = generated_answer.lower()
    if "true" in generated_answer or "false" in generated_answer:
        if "true" in generated_answer and "false" not in generated_answer:
            is_supported = True
        elif "false" in generated_answer and "true" not in generated_answer:
            is_supported = False
        else:
            is_supported = generated_answer.index("true") > generated_answer.index("false")
    else:
        is_supported = all([keyword not in generated_answer.translate(
                            str.maketrans("", "", string.punctuation)).split() for keyword in
                                            ["not", "cannot", "unknown", "information"]])
    return 'SUPPORTED' if is_supported else 'NOT_SUPPORTED'


def get_openai_prediction(response):
    log_probs = response.choices[0].logprobs.content[0].top_logprobs
    sorted_logprobs = sorted(log_probs, key=lambda x: x.logprob)  # sort ASC

    true_logprob = -np.inf
    false_logprob = -np.inf
    for log_prob in sorted_logprobs:
        token = log_prob.token.lower().strip()
        if token == 'true':
            true_logprob = log_prob.logprob
        elif token == 'false':
            false_logprob = log_prob.logprob

    if true_logprob != -np.inf or false_logprob != -np.inf:
        return 'SUPPORTED' if true_logprob > false_logprob else 'NOT_SUPPORTED'
    else:
        return 'UNKNOWN'
