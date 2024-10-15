"""Module for atomic fact loss."""

import torch
from torch import nn

from losses.custom_huber_loss import CustomHuberLoss


class AtomicFactsLoss(nn.Module):
    """Custom loss function for atomic fact prediction using Huber Loss."""
    def __init__(self, delta=0.66, huber_delta=0.3, pos_weight=1, neg_weight=1):
        """
        Initialize the AtomicFactsLoss class.

        :param delta: Threshold used for the loss calculation.
        :param huber_delta: Delta parameter for the Huber loss.
        :param pos_weight: Weight for positive class.
        :param neg_weight: Weight for negative class.
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.delta = delta
        self.huber_delta = huber_delta
        self.huber_loss = CustomHuberLoss(delta=huber_delta)

    def forward(self, y, atomic_preds, claim_mask):
        """
        Forward pass for computing the loss based on atomic facts and predictions.

        :param y: Ground truth labels.
        :param atomic_preds: Predictions for atomic facts.
        :param claim_mask: Mask to indicate which parts of the input are relevant for each example.
        :return: Computed loss.
        """
        a = (atomic_preds.unsqueeze(0) * claim_mask.unsqueeze(2)).squeeze(2)
        atomic_facts_count = a.count_nonzero(dim=1)

        # Loss for true facts
        b = a.masked_fill(torch.eq(a, 0), 1)
        loss_true = torch.relu(self.delta - b)
        loss_true = self.huber_loss(loss_true, torch.zeros_like(loss_true))
        loss_true = loss_true.sum(dim=1) / atomic_facts_count  # * atomic_facts_count

        # Loss for false facts
        min_atomic_preds = b.min(dim=1)[0]
        loss_false = torch.relu(min_atomic_preds - (1 - self.delta))
        loss_false = self.huber_loss(loss_false, torch.zeros_like(loss_false))

        # Combine losses
        loss = y * self.pos_weight * loss_true + (1 - y) * self.neg_weight * loss_false

        return loss.mean()
