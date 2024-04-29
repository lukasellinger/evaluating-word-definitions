"""Model for the second step of verifying a claim with a knowledge base.
Input: Claim to verify and sentences which should be used to verify the claim.
Output: SUPPORTS | REFUTES | NOT ENOUGH INFO
"""
from datetime import datetime

import torch
from torch import nn


class ClaimVerificationModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None,):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def save(self, name):
        timestamp = datetime.now().strftime("%m-%d_%H-%M")
        model_path = f'{name}_{timestamp}.pth'
        torch.save(self.state_dict(), model_path)