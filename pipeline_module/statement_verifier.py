"""Module for Statement Verifiers."""
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          pipeline)

from dataset.def_dataset import Fact
from models.claim_verification_model import ClaimVerificationModel


class StatementVerifier(ABC):
    """Abstract base class for verifying statements against evidence."""

    def __call__(self, statements: List[Dict], evidence_batch: List[List[Dict]]):
        """
        Verify a batch of statements against a batch of evidences.

        :param statements: List of statements to be verified.
        :param evidence_batch: List of evidences corresponding to the statements.
        :return: List of verification results.
        """
        return self.verify_statement_batch(statements, evidence_batch)

    @abstractmethod
    def set_premise_sent_order(self, sent_order: str):
        """
        Set the order in which premise sentences should be processed during verification.

        :param sent_order: The sentence order strategy ('reverse', 'top_last', or 'keep').
        """

    @abstractmethod
    def verify_statement(self, statement: Dict, evidence: List[Dict]):
        """
        Verify a single statement against a single evidence.

        :param statement: The statement to be verified.
        :param evidence: The evidence to verify the statement against.
        :return: Verification result.
        """

    @abstractmethod
    def verify_statement_batch(self,
                               statements: List[Dict], evids_batch: List[List[Dict]]) -> List[Dict]:
        """
        Verify a batch of statements against a batch of evidences.

        :param statements: List of statements to be verified.
        :param evids_batch: List of evidences corresponding to the statements.
        :return: List of verification results.
        """


class ModelStatementVerifier(StatementVerifier):
    """
    StatementVerifier implementation that uses a machine learning model for verification.
    """

    MODEL_NAME = 'lukasellinger/claim-verification-model-top_last'

    def __init__(self, model_name: str = '', premise_sent_order: str = 'top_last'):
        self.model_name = model_name or self.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.premise_sent_order = None
        self.set_premise_sent_order(premise_sent_order)

    def set_premise_sent_order(self, sent_order: str):
        if sent_order not in {'reverse', 'top_last', 'keep'}:
            raise ValueError(
                "premise_sent_order needs to be either 'reverse', 'top_last', or 'keep'")
        self.premise_sent_order = sent_order

    def load_model(self):
        """Load the machine learning model for verification, if not already loaded."""
        if self.model is None:
            model_raw = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model = ClaimVerificationModel(model_raw).to(self.device)
            self.model.eval()

    def unload_model(self):
        """Unload the machine learning model and free up GPU resources."""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def verify_statement(self, statement: Dict, evidence: List[Dict]):
        return self.verify_statement_batch([statement], [evidence])[0]

    def _order_hypothesis(self, hypo_sents: List[str]):
        if self.premise_sent_order not in {'reverse', 'top_last', 'keep'}:
            raise ValueError(
                "premise_sent_order needs to be either 'reverse', 'top_last', or 'keep'")

        if not hypo_sents:
            return ''

        if self.premise_sent_order == 'reverse':
            ordered_sents = hypo_sents[::-1]
        elif self.premise_sent_order == 'top_last':
            ordered_sents = hypo_sents[1:] + [hypo_sents[0]]
        else:  # 'keep'
            ordered_sents = hypo_sents

        return ' '.join(ordered_sents)

    def verify_statement_batch(self,
                               statements: List[Dict], evids_batch: List[List[Dict]]) -> List[Dict]:
        if not self.model:
            self.load_model()

        hypothesis_batch = [self._order_hypothesis([sentence['text'] for sentence in entry]) for
                            entry in evids_batch]
        predictions_batch = []
        for statement, hypothesis in zip(statements, hypothesis_batch):
            facts = statement.get('splits', [statement.get('text')])
            if not hypothesis:
                predictions = [Fact.NOT_SUPPORTED.name] * len(facts)
                factuality = Fact.NOT_SUPPORTED.to_factuality()
            else:
                model_inputs = self.tokenizer([hypothesis] * len(facts), facts,
                                              return_tensors='pt', padding=True).to(self.device)
                del model_inputs['token_type_ids']
                with torch.no_grad():
                    outputs = self.model(**model_inputs)
                    logits = outputs['logits']
                    probabilities = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probabilities, dim=-1).tolist()

                factuality = sum(pred == 0 for pred in predictions) / len(predictions)
                predictions = [Fact.SUPPORTED.name if pred == 0 else Fact.NOT_SUPPORTED.name for
                               pred in predictions]
            factualities = [{'atom': fact, 'predicted': prediction} for fact, prediction in
                            zip(facts, predictions)]

            predictions_batch.append({
                'predicted': Fact.SUPPORTED.name if factuality == 1 else Fact.NOT_SUPPORTED.name,
                'factuality': factuality,
                'atoms': factualities
            })
        return predictions_batch


class ModelEnsembleStatementVerifier(ModelStatementVerifier):
    """StatementVerifier implementation that uses aan ensemble of models for verification."""

    MODEL_NAME = 'lukasellinger/claim_verification_model-v5'

    def __init__(self, model_name: str = '', premise_sent_order: str = 'reverse'):
        super().__init__(model_name, premise_sent_order)
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli", device=self.device)

    def load_model(self):
        if self.model is None:
            model_raw = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model = ClaimVerificationModel(model_raw).to(self.device)
            self.model.eval()

    def unload_model(self):
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def verify_statement_batch(self,
                               statements: List[Dict], evids_batch: List[List[Dict]]) -> List[Dict]:
        if not self.model:
            self.load_model()

        hypothesis_batch = [self._order_hypothesis([sentence['text'] for sentence in entry]) for
                            entry in evids_batch]
        predictions_batch = []
        for statement, hypothesis in zip(statements, hypothesis_batch):
            facts = statement.get('splits', [statement.get('text')])
            if not hypothesis:
                predictions = [Fact.NOT_SUPPORTED.name] * len(facts)
                factuality = Fact.NOT_SUPPORTED.name
            else:
                model_inputs = self.tokenizer([hypothesis] * len(facts), facts,
                                              return_tensors='pt', padding=True).to(self.device)
                del model_inputs['token_type_ids']
                with torch.no_grad():
                    # Get logits from the initial model
                    outputs = self.model(**model_inputs)
                    logits = outputs['logits']
                    probabilities = torch.softmax(logits, dim=-1).cpu()
                    maximums, _ = torch.max(probabilities, dim=-1)

                    predictions = torch.argmax(probabilities, dim=-1).tolist()

                    # Identify low confidence cases (maximum probability < 0.1)
                    low_confidence_indices = (maximums < 0.7).nonzero(as_tuple=True)[0]

                    if len(low_confidence_indices) > 0:
                        # Prepare inputs for the low confidence cases
                        low_confidence_facts = [facts[i] for i in low_confidence_indices.tolist()]
                        candidate_labels = ['true', 'false']

                        outputs = self.classifier(low_confidence_facts, candidate_labels,
                                                  multi_label=True)
                        zero_shot_probs = torch.softmax(
                            torch.tensor([output.get('scores') for output in outputs]), dim=-1)
                        zero_shot_probs = torch.cat((zero_shot_probs, zero_shot_probs[:, 1:2]),
                                                    dim=1)
                        combined_probabilities = 1.5 * probabilities[
                            low_confidence_indices] + zero_shot_probs
                        combined_predictions = torch.argmax(combined_probabilities, dim=-1).tolist()
                        for i, idx in enumerate(low_confidence_indices):
                            predictions[idx] = combined_predictions[i]

                factuality = sum(pred == 0 for pred in predictions) / len(predictions)
                predictions = [Fact.SUPPORTED.name if pred == 0 else Fact.NOT_SUPPORTED.name for
                               pred in predictions]

            factualities = [{'atom': fact, 'predicted': prediction} for fact, prediction in
                            zip(facts, predictions)]

            predictions_batch.append({
                'predicted': Fact.SUPPORTED.name if factuality == 1 else Fact.NOT_SUPPORTED.name,
                'factuality': factuality,
                'atoms': factualities
            })

        return predictions_batch


if __name__ == "__main__":
    verifier = ModelStatementVerifier(premise_sent_order='top_last')
    results = verifier.verify_statement_batch(
        [{'text': 'Sun is hot.'}, {'text': 'Sun is cold.'}],
        [[{'text': 'Sun is very very very hot.'}],
         [{'text': 'Sun is very very very hot.'}]]
    )
    print(results)
