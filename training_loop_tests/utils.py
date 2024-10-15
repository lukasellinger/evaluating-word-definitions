"""Utils for training loop tests."""
from general_utils.utils import calc_bin_stats, plot_graph


def plot_stats(claim_lenghts, reference_lenghts, reference_label, gt_labels, pr_labels):
    """Plot stats for the training loop."""
    if len(claim_lenghts) > 0:
        bin_stats = calc_bin_stats(gt_labels, pr_labels, claim_lenghts)
        print(bin_stats)
        plot_graph(list(bin_stats.keys()), [entry['acc'] for entry in bin_stats.values()],
                   x_label='Claim Length', y_label='Acc')

    if len(claim_lenghts) > 0:
        bin_stats = calc_bin_stats(gt_labels, pr_labels, reference_lenghts)
        print(bin_stats)
        plot_graph(list(bin_stats.keys()), [entry['acc'] for entry in bin_stats.values()],
                   x_label=reference_label, y_label='Acc')
