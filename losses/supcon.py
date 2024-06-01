"""Module for supervised contrastive loss.
ref: https://arxiv.org/abs/2004.11362"""
import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature  # higher temperature leads to lower loss

    def forward(self, anchor: torch.tensor, references: torch.tensor, labels: torch.tensor) -> (
            torch.tensor):
        """
        Calculate the mean supervised contrastive loss over a batch. Each entry of the batch
        can have a different amount of positives and negatives. These are marked with the labels.
        Positives are marked with 1, Negatives with 0. Padding with -1.
        :param anchor: tensor of shape (b x 1 x d)
        :param references: tensor of shape (b x r x d)
        :param labels: tensor of shape (b x r)
        :return: mean batch loss
        """
        pos_count = torch.sum(torch.eq(labels, 1), dim=-1)

        similarity = cosine_similarity(anchor, references, dim=-1) / self.temperature
        logits_max = torch.max(similarity, dim=-1)[0].detach()
        similarity = similarity - logits_max.unsqueeze(-1)

        pos_pairs = similarity.masked_fill(~torch.eq(labels, 1), 0)
        sum_pos_pairs = torch.sum(pos_pairs, dim=-1)

        neg_pairs = similarity.masked_fill(~torch.eq(labels, 0), float('-inf'))
        pos_pairs = similarity.masked_fill(~torch.eq(labels, 1), float('-inf'))

        exp_pos_pairs = torch.exp(pos_pairs)
        exp_neg_pairs = torch.exp(neg_pairs)
        exp_total = torch.sum(exp_pos_pairs + exp_neg_pairs, dim=-1)
        log_exp_total = torch.log(exp_total)

        loss = - 1 / pos_count * (sum_pos_pairs - pos_count * log_exp_total)

        # In entries with more pos elements, the impact of each  pos element is smaller compared to
        # entries with fewer pos elements.
        return loss.mean()
