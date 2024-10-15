"""Module for Evidence Selector."""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

from general_utils.utils import rank_docs
from models.evidence_selection_model import EvidenceSelectionModel


class EvidenceSelector(ABC):
    """
    Abstract base class for selecting evidences.
    """

    def __call__(self, batch: List[Dict], evidence_batch: List[List[Dict]],
                 max_evidence_count: int = 3, top_k: int = 3) -> List[List[Dict]]:
        """
        Select evidences for a batch of claims by calling the select_evidences_batch method.

        :param batch: List of claims.
        :param evidence_batch: List of evidence lists corresponding to each claim.
        :param max_evidence_count: Maximum number of evidences to consider for each claim.
        :param top_k: Number of top sentences to select for each claim.
        :return: List of selected evidences for each claim.
        """
        return self.select_evidences_batch(batch, evidence_batch, max_evidence_count, top_k)

    @abstractmethod
    def set_min_similarity(self, min_similarity: float):
        """Set the minimum similarity threshold for evidence selection."""

    @abstractmethod
    def set_evidence_selection(self, evidence_selection: str):
        """
        Set the method of evidence selection.

        :param evidence_selection: Strategy for evidence selection ('mmr' or 'top').
        """

    @abstractmethod
    def select_evidences(self, claim: Dict, evidences: List[Dict]) -> List[Dict]:
        """
        Select evidences for a single claim from a list of evidences.

        :param claim: Dictionary representing the claim.
        :param evidences: List of evidence dictionaries.
        :return: List of selected evidences for the given claim.
        """

    @abstractmethod
    def select_evidences_batch(self, batch: List[Dict], evidence_batch: List[List[Dict]],
                               max_evidence_count: int = 3, top_k: int = 3) -> List[List[Dict]]:
        """
        Select evidences for a batch of claims from corresponding evidence batches.

        :param batch: List of claims.
        :param evidence_batch: List of evidence lists corresponding to each claim.
        :param max_evidence_count: Maximum number of evidences to consider for each claim.
        :param top_k: Number of top sentences to select for each claim.
        :return: List of selected evidences for each claim.
        """


class ModelEvidenceSelector(EvidenceSelector):
    """
    EvidenceSelector implementation that uses a machine learning model for evidence selection.
    """

    MODEL_NAME = 'lukasellinger/evidence-selection-model'

    def __init__(self,
                 model_name: str = '', min_similarity: float = 0, evidence_selection: str = 'top'):
        """
        Initialize the ModelEvidenceSelector with the specified model.

        :param model_name: Name of the model to use. Defaults to a pre-defined model.
        """
        self.model_name = model_name or self.MODEL_NAME
        self.min_similarity = min_similarity
        self.evidence_selection = evidence_selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

    def set_min_similarity(self, min_similarity: float):
        self.min_similarity = min_similarity

    def set_evidence_selection(self, evidence_selection: str):
        if evidence_selection not in ['mmr', 'top']:
            raise ValueError('evidence_selection must be either mmr or top.')
        self.evidence_selection = evidence_selection

    def load_model(self):
        """Load the machine learning model for evidence selection, if not already loaded."""
        if self.model is None:
            model_raw = AutoModel.from_pretrained(self.model_name, trust_remote_code=True,
                                                  add_pooling_layer=False, safe_serialization=True)
            self.model = EvidenceSelectionModel(model_raw).to(self.device)
            self.model.eval()

    def unload_model(self):
        """Unload the machine learning model and free up GPU resources."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def select_evidences(self, claim: Dict, evidences: List[Dict]) -> List[Dict]:
        return self.select_evidences_batch([claim], [evidences])[0]

    def select_evidences_batch(self, batch: List[Dict],
                               evidence_batch: List[List[Dict]],
                               max_evidence_count: int = 3, top_k: int = 3) -> List[List[Dict]]:
        if not self.model:
            self.load_model()

        ranked_evidence_batch = self._rank_evidences(batch, evidence_batch, max_evidence_count)
        top_sentences_batch = self._select_top_sentences(batch, ranked_evidence_batch, top_k)
        return top_sentences_batch

    def _rank_evidences(self, batch: List[Dict], evidence_batch: List[List[Dict]],
                        max_evidence_count: int) -> List[List[Dict]]:
        ranked_evidence_batch = []
        for claim, evidences in zip(batch, evidence_batch):
            if len(evidences) > max_evidence_count:
                ranked_indices = rank_docs(claim['text'],
                                           [" ".join(evidence.get('lines')) for evidence in
                                            evidences], k=max_evidence_count)
                ranked_evidence_batch.append([evidences[i] for i in ranked_indices])
            else:
                ranked_evidence_batch.append(evidences)
        return ranked_evidence_batch

    @staticmethod
    def mmr(sentence_similarities: List[Dict],
            top_n: int =3, lambda_param: float = 0.7) -> List[Dict]:
        """
        Apply Maximal Marginal Relevance (MMR) to select the top_n sentences based on relevance
        and diversity.

        :param sentence_similarities: List of sentence similarity scores and embeddings.
        :param top_n: Number of top sentences to select.
        :param lambda_param: Parameter for controlling the trade-off between relevance and
        diversity.
        :return: List of selected sentences after applying MMR.
        """
        if len(sentence_similarities) < 1:
            return []

        claim_similarities = [entry['sim'] for entry in sentence_similarities]
        sentence_embeddings = torch.stack([entry['embedding']
                                           for entry in sentence_similarities]).cpu()

        pairwise_similarities = cosine_similarity(
            sentence_embeddings.unsqueeze(1), sentence_embeddings.unsqueeze(0), dim=2
        )

        selected_indices = []
        candidate_indices = list(range(len(sentence_embeddings)))

        for _ in range(min(top_n, len(candidate_indices))):
            mmr_scores = []
            for i in candidate_indices:
                relevance = claim_similarities[i]

                diversity = 0
                if selected_indices:
                    diversity = max(pairwise_similarities[i][j] for j in selected_indices)

                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append(mmr_score)

            best_index = candidate_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_index)
            candidate_indices.remove(best_index)

        selected_elements = [sentence_similarities[i] for i in selected_indices]
        return selected_elements

    def _select_top_sentences(self, batch: List[Dict], ranked_evidence_batch: List[List[Dict]],
                              top_k: int) -> List[List[Dict]]:
        top_sentences_batch = []
        for claim, evidences in zip(batch, ranked_evidence_batch):
            statement_model_input = self.tokenizer(claim['text'], return_tensors='pt').to(
                self.device)
            with torch.no_grad():
                statement_embeddings = self.model(**statement_model_input)
            sentence_similarities = []

            for entry in evidences:
                sentence_similarities.extend(
                    self._compute_sentence_similarities(entry['title'], entry['line_indices'],
                                                        entry['lines'], statement_embeddings))

            filtered_sentences = filter(lambda x: x['sim'] > self.min_similarity,
                                        sentence_similarities)
            sorted_sentences = sorted(filtered_sentences, key=lambda x: x['sim'], reverse=True)
            if self.evidence_selection == 'mmr':
                top_sentences = self.mmr(sorted_sentences, top_k)
            elif self.evidence_selection == 'top':
                top_sentences = self.get_top_unique_sentences(sorted_sentences, top_k)
            else:
                raise ValueError('evidence_selection must either be "mmr" or "top"')

            for entry in top_sentences:
                entry.pop('embedding', None)
            top_sentences_batch.append(top_sentences)
        return top_sentences_batch

    def _compute_sentence_similarities(self,
                                       page: str, line_numbers: List[str], sentences: List[str],
                                       statement_embeddings: torch.Tensor) -> List[Dict]:
        encoded_sequence, sentence_mask = self._encode_sentences(sentences)
        sentences_model_input = {
            'input_ids': torch.tensor(encoded_sequence).unsqueeze(0).to(self.device),
            'attention_mask': torch.ones(len(encoded_sequence)).unsqueeze(0).to(self.device),
            'sentence_mask': torch.tensor(sentence_mask).unsqueeze(0).to(self.device)
        }
        with torch.no_grad():
            sentence_embeddings = self.model(**sentences_model_input).squeeze(0)
            claim_similarities = cosine_similarity(statement_embeddings,
                                                   sentence_embeddings, dim=2).tolist()[0]
        return [{'title': page,
                 'line_idx': line_num,
                 'text': sentence,
                 'sim': sim,
                 'embedding': embedding} for line_num, sentence, sim, embedding in
                zip(line_numbers, sentences, claim_similarities, sentence_embeddings)]

    def _encode_sentences(self, sentences: List[str]) -> Tuple[List[int], List[List[int]]]:
        encoded_sequence = []
        sentence_mask = []
        for i, sentence in enumerate(sentences):
            encoded_sentence = self.tokenizer.encode(sentence)[1:-1]
            encoded_sequence += encoded_sentence
            sentence_mask += [i] * len(encoded_sentence)
            encoded_sequence.append(self.tokenizer.sep_token_id)
            sentence_mask.append(-1)
        unique_sentence_numbers = set(sentence_mask)
        sentence_masks = [[1 if val == num else 0 for val in sentence_mask] for num in
                          unique_sentence_numbers if num != -1]
        return encoded_sequence, sentence_masks

    @staticmethod
    def get_top_unique_sentences(sorted_sentences: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Select the top K unique sentences based on similarity scores.

        :param sorted_sentences: List of sorted sentences based on similarity scores.
        :param top_k: Number of top unique sentences to select.
        :return: List of top K unique sentences.
        """
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
        [{'text': 'sun is shining.'}],
        [[{'title': 'Page1', 'line_idx': '0', 'text': 'Sun was good'},
          ('Page1', '1', 'shining light')]]
    ))
