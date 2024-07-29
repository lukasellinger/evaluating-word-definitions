from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

from general_utils.utils import rank_docs
from models.evidence_selection_model import EvidenceSelectionModel


class EvidenceSelector(ABC):
    """
    Abstract base class for selecting evidences.
    """

    def __call__(self, batch: List[Dict], evidence_batch: List[List[Dict]]):
        """
        Select evidences for a batch of claims.

        :param batch: List of claims.
        :param evidence_batch: List of evidence lists corresponding to each claim.
        :return: List of selected evidences for each claim.
        """
        return self.select_evidences_batch(batch, evidence_batch)

    @abstractmethod
    def select_evidences(self, claim: Dict, evidences: List[Dict]):
        """
        Select evidences for a single claim.

        :param claim: The claim to select evidences for.
        :param evidences: List of evidences.
        :return: List of selected evidences.
        """
        pass

    @abstractmethod
    def select_evidences_batch(self, batch: List[Dict], evidence_batch: List[List[Dict]]):
        """
        Select evidences for a batch of claims.

        :param batch: List of claims.
        :param evidence_batch: List of evidence lists corresponding to each claim.
        :return: List of selected evidences for each claim.
        """
        pass


class ModelEvidenceSelector(EvidenceSelector):
    """
    EvidenceSelector implementation that uses a machine learning model for evidence selection.
    """

    MODEL_NAME = 'lukasellinger/evidence_selection_model-v2'

    def __init__(self, model_name: str = ''):
        """
        Initialize the ModelEvidenceSelector with the specified model.

        :param model_name: Name of the model to use. Defaults to a pre-defined model.
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

    def load_model(self):
        if self.model is None:
            model_raw = AutoModel.from_pretrained(self.model_name, trust_remote_code=True,
                                                  add_pooling_layer=False, safe_serialization=True)
            self.model = EvidenceSelectionModel(model_raw).to(self.device)
            self.model.eval()

    def unload_model(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def select_evidences(self, claim: Dict, evidences: List[Tuple[str, str, str]]):
        """
        Select evidences for a single claim.

        :param claim: The claim to select evidences for.
        :param evidences: List of evidences.
        :return: List of selected evidences.
        """
        return self.select_evidences_batch([claim], [evidences])[0]

    def select_evidences_batch(self, batch: List[Dict], evidence_batch: List[List[Tuple[str, str, str]]], max_evidence_count: int = 3, top_k: int = 3):
        """
        Select evidences for a batch of claims.

        :param batch: List of claims.
        :param evidence_batch: List of evidence lists corresponding to each claim.
        :param max_evidence_count: Maximum number of evidences to consider.
        :param top_k: Number of top sentences to select.
        :return: List of selected evidences for each claim.
        """
        if not self.model:
            self.load_model()

        ranked_evidence_batch = self._rank_evidences(batch, evidence_batch, max_evidence_count)
        top_sentences_batch = self._select_top_sentences(batch, ranked_evidence_batch, top_k)
        return top_sentences_batch

    def _rank_evidences(self, batch: List[Dict], evidence_batch: List[List[Tuple[str, str, str]]], max_evidence_count: int):
        ranked_evidence_batch = []
        for claim, evidences in zip(batch, evidence_batch):
            if len(evidences) > max_evidence_count:
                ranked_indices = rank_docs(claim['text'], [" ".join(evidence.get('lines')) for evidence in evidences], k=max_evidence_count)
                ranked_evidence_batch.append([evidences[i] for i in ranked_indices])
            else:
                ranked_evidence_batch.append(evidences)
        return ranked_evidence_batch

    def _select_top_sentences(self, batch: List[Dict], ranked_evidence_batch: List[List[Dict]], top_k: int):
        top_sentences_batch = []
        for claim, evidences in zip(batch, ranked_evidence_batch):
            statement_model_input = self.tokenizer(claim['text'], return_tensors='pt').to(self.device)
            with torch.no_grad():
                statement_embeddings = self.model(**statement_model_input)

            sentence_similarities = []
            for entry in evidences:
                sentence_similarities.extend(self._compute_sentence_similarities(entry['title'], entry['line_indices'], entry['lines'], statement_embeddings))

            sorted_sentences = sorted(sentence_similarities, key=lambda x: x['sim'], reverse=True)
            top_sentences = self._get_top_unique_sentences(sorted_sentences, top_k)
            top_sentences_batch.append(top_sentences)
        return top_sentences_batch

    def _compute_sentence_similarities(self, page: str, line_numbers: List[str], sentences: List[str], statement_embeddings: torch.Tensor):
        encoded_sequence, sentence_mask = self._encode_sentences(sentences)
        sentences_model_input = {
            'input_ids': torch.tensor(encoded_sequence).unsqueeze(0).to(self.device),
            'attention_mask': torch.ones(len(encoded_sequence)).unsqueeze(0).to(self.device),
            'sentence_mask': torch.tensor(sentence_mask).unsqueeze(0).to(self.device)
        }
        with torch.no_grad():
            sentence_embeddings = self.model(**sentences_model_input)
            claim_similarities = cosine_similarity(statement_embeddings, sentence_embeddings, dim=2).tolist()[0]
        return [{'title': page, 'line_idx': line_num, 'text': sentence, 'sim': sim} for line_num, sentence, sim in zip(line_numbers, sentences, claim_similarities)]

    def _encode_sentences(self, sentences: List[str]):
        encoded_sequence = []
        sentence_mask = []
        for i, sentence in enumerate(sentences):
            encoded_sentence = self.tokenizer.encode(sentence)[1:-1]
            encoded_sequence += encoded_sentence
            sentence_mask += [i] * len(encoded_sentence)
            encoded_sequence.append(self.tokenizer.sep_token_id)
            sentence_mask.append(-1)
        unique_sentence_numbers = set(sentence_mask)
        sentence_masks = [[1 if val == num else 0 for val in sentence_mask] for num in unique_sentence_numbers if num != -1]
        return encoded_sequence, sentence_masks

    @staticmethod
    def _get_top_unique_sentences(sorted_sentences: List[tuple], top_k: int = 3):
        unique_sentences = []
        seen_sentences = set()
        for entry in sorted_sentences:
            if entry['text'] not in seen_sentences:
                unique_sentences.append(entry)
                seen_sentences.add(entry['text'])
            if len(unique_sentences) == top_k:
                break
        return unique_sentences


if __name__ == "__main__":
    selector = ModelEvidenceSelector()
    print(selector.select_evidences_batch(
        [{'text': 'sun is shining.'}, {'text': 'I like it.'}],
        [[('Page1', '0', 'Sun was good'), ('Page1', '1', 'shining light'), ('Page1', '2', 'Hi, how are you'), ('Page1', '3', 'Yes of course')],
         [('Page2', '0', 'I hate you.')]]
    ))
