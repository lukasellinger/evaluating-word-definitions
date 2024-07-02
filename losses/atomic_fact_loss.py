"""Module for atomic fact loss."""

import torch
import torch.nn as nn

from losses.custom_huber_loss import CustomHuberLoss


class AtomicFactsLoss(nn.Module):
    def __init__(self, delta=0.66, huber_delta=0.3):
        super(AtomicFactsLoss, self).__init__()
        self.delta = delta
        self.huber_delta = huber_delta
        self.huber_loss = CustomHuberLoss(delta=huber_delta)

    def forward(self, y, atomic_preds, claim_mask):
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
        loss = y * loss_true + (1 - y) * loss_false

        return loss.mean()
