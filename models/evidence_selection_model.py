"""Model for the first step of verifying a claim with a knowledge base.
Input: Claim to verify and document which should be used for verifying.
Output: Sentence Embeddings.
"""
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F


class EvidenceSelectionModel(nn.Module):
    def __init__(self, model, feed_forward=False, normalize_before_fc=True, out_features=256):
        super().__init__()
        self.model = model
        self.feed_forward = feed_forward
        self.normalize_before_fc = normalize_before_fc

        if self.feed_forward:
            for param in self.model.parameters():
                param.requires_grad = False
            self.fc = nn.Linear(1024, out_features)

    def forward(self, input_ids=None, attention_mask=None, sentence_mask=None):
        if sentence_mask is None:
            # keep in mind that here cls and end token are inside the mask.
            sentence_mask = attention_mask.unsqueeze(dim=1)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']

        if self.feed_forward:
            if self.normalize_before_fc:
                outputs = F.normalize(outputs, dim=2)
            outputs = self.fc(outputs)
        return self.sentence_mean_pooling(outputs, sentence_mask)

    @staticmethod
    def sentence_mean_pooling(model_output, sentence_mask):
        token_embeddings = model_output.unsqueeze(1)

        masks_size = sentence_mask.count_nonzero(dim=-1)
        masks = sentence_mask.unsqueeze(-1)

        sentence_embeddings = (masks * token_embeddings).sum(dim=2) / masks_size.unsqueeze(-1)
        return sentence_embeddings

    def save(self, name):
        timestamp = datetime.now().strftime("%m-%d_%H-%M")
        model_path = f'{name}_{timestamp}.pth'
        torch.save(self.state_dict(), model_path)
