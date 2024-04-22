"""Model for the first step of verifying a claim with a knowledge base.
Input: Claim to verify and document which should be used for verifying.
Output: Top N most similar sentences to the claim.
"""
import torch
from torch import nn
import torch.nn.functional as F


class EvidenceSelectionModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, sentence_mask=None):
        if sentence_mask is None:
            sentence_mask = torch.ones(input_ids.shape)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings, unique_sents = self.sentence_mean_pooling(outputs, sentence_mask)
        return embeddings, unique_sents

    @staticmethod
    def sentence_mean_pooling(model_output, sentence_mask):
        token_embeddings = model_output['last_hidden_state'].unsqueeze(1)

        unique_sents = torch.unique(sentence_mask)
        masks = torch.stack([(sentence_mask == num) for num in unique_sents])
        masks = masks.permute(1, 0, 2)
        masks_size = masks.count_nonzero(dim=2)
        masks = masks.unsqueeze(-1)

        sentence_embeddings = (masks * token_embeddings).sum(dim=2) / masks_size.unsqueeze(-1)
        # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings, unique_sents
