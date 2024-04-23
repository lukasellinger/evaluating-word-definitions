"""Model for the second step of verifying a claim with a knowledge base.
Input: Claim to verify and sentences which should be used to verify the claim.
Output: SUPPORTS | REFUTES | NOT ENOUGH INFO
"""

from torch import nn


class ClaimVerificationModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None,):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
