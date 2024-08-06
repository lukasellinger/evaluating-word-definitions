from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset.def_dataset import Fact
from models.claim_verification_model import ClaimVerificationModel


class StatementVerifier(ABC):
    """
    Abstract base class for verifying statements against evidence.
    """

    def __call__(self, statements: List[str], evidence_batch: List[str]):
        """
        Verify a batch of statements against a batch of evidences.

        :param statements: List of statements to be verified.
        :param evidence_batch: List of evidences corresponding to the statements.
        :return: List of verification results.
        """
        return self.verify_statement_batch(statements, evidence_batch)

    @abstractmethod
    def verify_statement(self, statement: Dict, evidence: str):
        """
        Verify a single statement against a single evidence.

        :param statement: The statement to be verified.
        :param evidence: The evidence to verify the statement against.
        :return: Verification result.
        """
        pass

    @abstractmethod
    def verify_statement_batch(self, statements: List[Dict], evids_batch: List[str]):
        """
        Verify a batch of statements against a batch of evidences.

        :param statements: List of statements to be verified.
        :param evids_batch: List of evidences corresponding to the statements.
        :return: List of verification results.
        """
        pass


class ModelStatementVerifier(StatementVerifier):
    """
    StatementVerifier implementation that uses a machine learning model for verification.
    """

    MODEL_NAME = 'lukasellinger/claim_verification_model-v1'

    def __init__(self, model_name: str = ''):
        """
        Initialize the ModelStatementVerifier with the specified model.

        :param model_name: Name of the model to use. Defaults to a pre-defined model.
        """
        self.model_name = model_name or self.MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

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

    def verify_statement(self, statement: Dict, evidence: List[str]):
        """
        Verify a single statement against a single evidence.

        :param statement: The statement to be verified.
        :param evidence: The evidence to verify the statement against.
        :return: Verification result.
        """
        return self.verify_statement_batch([statement], [evidence])[0]

    def verify_statement_batch(self, statements: List[Dict], evids_batch: List[List[str]]):
        """
        Verify a batch of statements against a batch of evidences.

        :param statements: List of statements to be verified.
        :param evids_batch: List of evidences corresponding to the statements.
        :return: List of verification results.
        """
        if not self.model:
            self.load_model()

        hypothesis_batch = [' '.join([sentence['text'] for sentence in entry[::-1]]) for entry in evids_batch]

        predictions_batch = []
        for statement, hypothesis in zip(statements, hypothesis_batch):
            facts = statement.get('splits', [statement.get('text')])
            model_inputs = self.tokenizer([hypothesis] * len(facts), facts,
                                          return_tensors='pt', padding=True).to(self.device)
            del model_inputs['token_type_ids']
            with torch.no_grad():
                outputs = self.model(**model_inputs)
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1).tolist()

            factuality = sum(pred == 0 for pred in predictions) / len(predictions)
            predictions = [Fact.SUPPORTED.name if pred == 0 else Fact.NOT_SUPPORTED.name for pred in predictions]
            factualities = [{'atom': fact, 'predicted': prediction} for fact, prediction in zip(facts, predictions)]

            predictions_batch.append({
                'predicted': Fact.SUPPORTED.name if factuality == 1 else Fact.NOT_SUPPORTED.name,
                'factuality': factuality,
                'atoms': factualities
            })
        return predictions_batch


if __name__ == "__main__":
    verifier = ModelStatementVerifier()
    results = verifier.verify_statement_batch(
        [{'text': 'Sun is hot.'}, {'text': 'Sun is cold.'}],
        [['Sun is very very very hot.'], ['Sun is very very very hot.']]
    )
    print(results)
