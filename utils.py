"""General utils for processing."""
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score, f1_score


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
