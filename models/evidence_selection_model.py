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

        sentence_embeddings = self.sentence_mean_pooling(outputs, sentence_mask)
        if self.feed_forward:
            if self.normalize_before_fc:
                sentence_embeddings = F.normalize(sentence_embeddings, dim=2)
            return self.fc(sentence_embeddings)
        else:
            return sentence_embeddings

    @staticmethod
    def sentence_mean_pooling(model_output, masks):
        model_output.unsqueeze_(1)

        masks_size = masks.count_nonzero(dim=-1)
        masks.unsqueeze_(-1)

        masks_size = torch.clamp(masks_size, min=1e-9)  # do not divide by 0
        sentence_embeddings = (masks * model_output).sum(dim=2) / masks_size.unsqueeze(-1)
        return sentence_embeddings

    def save(self, name):
        # model_path = f'{name}.pth'
        if self.feed_forward:
            torch.save(self.fc.state_dict(), f'{name}_fc.pth')
        else:
            self.model.save_pretrained(f'{name}')
