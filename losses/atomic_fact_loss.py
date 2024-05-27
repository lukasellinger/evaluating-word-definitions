"""Module for atomic fact loss."""

import torch
import torch.nn as nn


class AtomicFactsLoss(nn.Module):
    def __init__(self, delta=0.5):
        super(AtomicFactsLoss, self).__init__()
        self.delta = delta
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, y, atomic_preds, claim_mask):
        a = (atomic_preds.unsqueeze(0) * claim_mask.unsqueeze(2)).squeeze(2)

        atomic_facts_count = a.count_nonzero(dim=1)

        # Loss for true facts
        #loss_true = -torch.log(atomic_preds.squeeze(1)) * labels
        b = a.masked_fill(torch.eq(a, 0), 1)
        loss_true = -torch.log(b)
        loss_true = loss_true.sum(dim=1) / atomic_facts_count
        #mean_loss_true = loss_true.sum() / labels.sum()

        # Loss for false facts
        min_atomic_preds = b.min(dim=1)[0]
        loss_false = torch.relu(min_atomic_preds - self.delta)

        # Combine losses
        loss = y * loss_true + (1 - y) * loss_false

        return loss.mean()
