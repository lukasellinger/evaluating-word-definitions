"""Model for the first step of verifying a claim with a knowledge base.
Input: Claim to verify and document which should be used for verifying.
Output: Sentence Embeddings.
"""
from torch import nn


class EvidenceSelectionModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, sentence_mask=None):
        if sentence_mask is None:
            # keep in mind that here cls and end token are inside the mask.
            sentence_mask = attention_mask.unsqueeze(dim=1)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.sentence_mean_pooling(outputs, sentence_mask)

    @staticmethod
    def sentence_mean_pooling(model_output, sentence_mask):
        token_embeddings = model_output['last_hidden_state'].unsqueeze(1)

        masks_size = sentence_mask.count_nonzero(dim=-1)
        masks = sentence_mask.unsqueeze(-1)

        sentence_embeddings = (masks * token_embeddings).sum(dim=2) / masks_size.unsqueeze(-1)
        return sentence_embeddings
