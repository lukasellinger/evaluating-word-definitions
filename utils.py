"""General utils for processing."""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")

    raise ValueError(f"Unsupported string type: {type(text)}")


def calc_bin_stats(gt_labels, pr_labels, values):
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
            acc = accuracy_score(bin_gt_labels, bin_pr_labels)
            f1_weighted = f1_score(bin_gt_labels, bin_pr_labels, average='weighted')
            f1_macro = f1_score(bin_gt_labels, bin_pr_labels, average='macro')
            bin_stats[bin_upper] = {'acc': acc, 'f1_weighted': f1_weighted, 'f1_macro': f1_macro}

    return bin_stats


def plot_graph(keys, values, x_label='', y_label='', title=''):
    plt.plot(keys, values, marker='o', linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(keys, rotation=45)
    plt.show()
