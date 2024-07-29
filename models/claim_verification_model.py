"""Model for the second step of verifying a claim with a knowledge base.
Input: Claim to verify and sentences which should be used to verify the claim.
Output: SUPPORTS | REFUTES | NOT ENOUGH INFO
"""
from torch import nn


class ClaimVerificationModel(nn.Module):
    """Model to verify a claim. Hypotheses, Premise style."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        """Forward function."""
        return self.model(**kwargs)

    def save(self, name):
        """Stores the model."""
        self.model.save_pretrained(f'{name}')
