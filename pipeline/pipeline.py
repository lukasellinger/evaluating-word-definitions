"""Pipelines for the claim verification process."""
from typing import Type

import torch
from transformers import BigBirdModel, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from database.db_retriever import FeverDocDB
from dataset.def_dataset import Fact, process_sentence, process_lines
from models.claim_verification_model import ClaimVerificationModel
from models.evidence_selection_model import EvidenceSelectionModel


class Pipeline:
    """General Pipeline. Implement fetch_evidence, select_evidence, verify_claim."""

    def verify(self, word: str, claim: str) -> float:
        """
        Verify a claim related to a word. TODO: use numpy
        :param word: Word associated to the claim.
        :param claim: Claim to be verified.
        :return: percentage of true facts inside the claim.
        """
        ev_sents = self.fetch_evidence(word)
        atomic_claims = self.process_claim(claim)

        factuality = 0
        for atomic_claim in atomic_claims:
            selected_ev_sents = self.select_evidence(atomic_claim, ev_sents)
            factuality += self.verify_claim(atomic_claim, selected_ev_sents)

        return factuality / len(atomic_claims)

    @staticmethod
    def process_claim(claim: str) -> list[str]:
        """Process a claim. E.g. split it into its atomic facts."""
        return [claim]

    def fetch_evidence(self, word: str) -> list[str]:
        """
        Fetch the information of the word inside the knowledge base.
        :param word: Word, for which we need information.
        :return: List of sentences, representing all information known to the word.
        """

    def select_evidence(self, claim: str, sentences: list[str]) -> list[str]:
        """
        Select sentences possibly containing evidence for the claim.
        :param claim: Claim to be verified.
        :param sentences: Sentences to choose from.
        :return: List of sentences, possibly containing evidence.
        """

    def verify_claim(self, claim: str, sentences: list[str]) -> int:
        """
        Verify the claim using sentences as evidence.
        :param claim: Claim to be verified.
        :param sentences: Sentences to use as evidence.
        :return: 1, if claim can be verified, else 0.
        """


class TestPipeline(Pipeline):
    """Pipeline used for test purposes."""

    def __init__(self, selection_model=None, selection_model_tokenizer=None,
                 verification_model=None, verification_model_tokenizer=None):
        super(TestPipeline).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not selection_model:
            model_name = 'google/bigbird-roberta-large'
            model = BigBirdModel.from_pretrained(model_name)
            selection_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
            selection_model = EvidenceSelectionModel(model).to(self.device)
        self.selection_model = selection_model
        self.selection_model_tokenizer = selection_model_tokenizer

        if not verification_model:
            model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            verification_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            verification_model = ClaimVerificationModel(model).to(self.device)
        self.verification_model = verification_model
        self.verification_model_tokenizer = verification_model_tokenizer

    def fetch_evidence(self, word: str) -> list[str]:
        with FeverDocDB() as db:
            lines = db.get_doc_lines(word)

        lines = process_lines(lines)
        processed_lines = []
        for line in lines.split('\n'):
            line = process_sentence(line)
            line_number = line.split('\t')[0]
            line = line.lstrip(f'{line_number}\t')
            processed_lines.append(line)
        return processed_lines

    def select_evidence(self, claim: str, sentences: list[str], top_k=3) -> list[str]:
        claim_model_input, sentences_model_input = self.build_selection_model_input(claim, sentences)
        with torch.no_grad():
            claim_embedding = self.selection_model(**claim_model_input)
            sentence_embeddings = self.selection_model(**sentences_model_input)
            claim_similarities = F.cosine_similarity(claim_embedding, sentence_embeddings, dim=2)
            top_indices = torch.topk(claim_similarities, k=top_k)[1].squeeze(0)  # TODO does not work with batch

        return [sentences[index] for index in top_indices]

    def build_selection_model_input(self, claim: str, sentences: list[str]):
        encoded_sequence = []
        sentence_mask = []
        for i, sentence in enumerate(sentences):
            encoded_sentence = self.selection_model_tokenizer.encode(sentence)[1:-1]  # + [1]
            encoded_sequence += encoded_sentence
            sentence_mask += [i] * len(encoded_sentence)
            encoded_sequence.append(self.selection_model_tokenizer.sep_token_id)
            sentence_mask.append(-1)

        unique_sentence_numbers = set(sentence_mask)
        sentence_masks = []
        for num in unique_sentence_numbers:
            if num == -1:
                continue
            sentence_masks.append([1 if val == num else 0 for val in sentence_mask])

        return (self.selection_model_tokenizer(claim, return_tensors='pt'),
                {'input_ids': torch.tensor(encoded_sequence).unsqueeze(0),
                 'attention_mask': torch.ones(len(encoded_sequence)).unsqueeze(0),
                 'sentence_mask': torch.tensor(sentence_masks).unsqueeze(0)})

    def verify_claim(self, claim: str, sentences: list[str]) -> Fact:
        model_inputs = self.build_verification_model_input(claim, sentences)
        with torch.no_grad():
            output = self.verification_model(**model_inputs)
            predicted = torch.softmax(output['logits'], dim=-1)
            predicted = torch.argmax(predicted, dim=-1).item()
        return Fact(predicted).to_factuality()

    def build_verification_model_input(self, claim: str, sentences: list[str]):
        hypothesis = ' '.join(sentences)
        model_inputs = self.verification_model_tokenizer(hypothesis, claim)

        return {'input_ids': torch.tensor(model_inputs['input_ids']).unsqueeze(0),
                'attention_mask': torch.tensor(model_inputs['attention_mask']).unsqueeze(0)}

class WikiPipeline(Pipeline):
    """Pipeline using Wikipedia."""

    def __init__(self):
        super(WikiPipeline).__init__()


if __name__ == "__main__":
    pipeline = TestPipeline()
    print(pipeline.verify(word='Newfoundland_and_Labrador',
                          claim='Newfoundland and Labrador is the most linguistically homogeneous of Canada.'))
