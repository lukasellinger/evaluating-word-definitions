from abc import ABC, abstractmethod
from typing import List, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset.def_dataset import Fact
from fetchers.openai import OpenAiFetcher
from general_utils.utils import get_openai_prediction, parse_model_answer, \
    get_openai_prediction_log_probs
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
    def set_premise_sent_order(self, sent_order: str):
        pass

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

    MODEL_NAME = 'lukasellinger/claim_verification_model-v5'

    def __init__(self, model_name: str = '', premise_sent_order: str = 'reverse'):
        """
        Initialize the ModelStatementVerifier with the specified model.

        :param model_name: Name of the model to use. Defaults to a pre-defined model.
        """
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

    def _order_hypothesis(self, hypo_sents: List[str]):
        if self.premise_sent_order not in {'reverse', 'top_last', 'keep'}:
            raise ValueError(
                "premise_sent_order needs to be either 'reverse', 'top_last', or 'keep'")
        if len(hypo_sents) == 0:
            return ''

        if self.premise_sent_order == 'reverse':
            return ' '.join([sentence for sentence in hypo_sents[::-1]])
        elif self.premise_sent_order == 'top_last':
            return ' '.join([sentence for sentence in hypo_sents[1:] + [hypo_sents[0]]])
        elif self.premise_sent_order == 'keep':
            return ' '.join(hypo_sents)

    def verify_statement_batch(self, statements: List[Dict], evids_batch: List[List[str]]):
        """
        Verify a batch of statements against a batch of evidences.

        :param statements: List of statements to be verified.
        :param evids_batch: List of evidences corresponding to the statements.
        :return: List of verification results.
        """
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
    """
    StatementVerifier implementation that uses a machine learning model for verification.
    """

    MODEL_NAME = 'lukasellinger/claim_verification_model-v5'

    def __init__(self, model_name: str = '', premise_sent_order: str = 'reverse'):
        """
        Initialize the ModelStatementVerifier with the specified model.

        :param model_name: Name of the model to use. Defaults to a pre-defined model.
        """
        super().__init__(model_name, premise_sent_order)
        from transformers import pipeline
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli")
        self.openai = OpenAiFetcher()

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

    def verify_statement_batch(self, statements: List[Dict], evids_batch: List[List[str]]):
        """
        Verify a batch of statements against a batch of evidences.

        :param statements: List of statements to be verified.
        :param evids_batch: List of evidences corresponding to the statements.
        :return: List of verification results.
        """
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
                    probabilities = torch.softmax(logits, dim=-1)
                    maximums, _ = torch.max(probabilities, dim=-1)

                    predictions = torch.argmax(probabilities, dim=-1).tolist()

                    # Identify low confidence cases (maximum probability < 0.1)
                    low_confidence_indices = (maximums < 0.7).nonzero(as_tuple=True)[0]

                    if len(low_confidence_indices) > 0:
                        # # Prepare inputs for the low confidence cases
                        # low_confidence_facts = [facts[i] for i in low_confidence_indices.tolist()]
                        # for i, low_fact in zip(low_confidence_indices, low_confidence_facts):
                        #     response = self.openai.get_output([
                        #         {
                        #             "role": "user",
                        #             "content": f"Please verify the following statement. Input: {low_fact} True or False?\nOutput:'"
                        #         }
                        #     ])
                        #
                        #     true_log_prob, false_log_prob = get_openai_prediction_log_probs(
                        #         response, batched=False)
                        #     zero_shot_probs = torch.softmax(
                        #         torch.tensor([true_log_prob, false_log_prob]), dim=-1)
                        #     zero_shot_probs = torch.cat((zero_shot_probs, zero_shot_probs[1:2]),
                        #                                 dim=-1)
                        #     probabilities[i] = 2.5 * probabilities[i] + zero_shot_probs  # 1.5
                        # predictions = torch.argmax(probabilities, dim=-1).tolist()
                        # Prepare inputs for the low confidence cases
                        low_confidence_facts = [facts[i] for i in low_confidence_indices.tolist()]
                        candidate_labels = ['true', 'false']

                        outputs = self.classifier(low_confidence_facts, candidate_labels, multi_label=True)
                        zero_shot_probs = torch.softmax(torch.tensor([output.get('scores') for output in outputs]), dim=-1)
                        zero_shot_probs = torch.cat((zero_shot_probs, zero_shot_probs[:, 1:2]), dim=1)
                        combined_probabilities = 1.5 * probabilities[low_confidence_indices] + zero_shot_probs
                        combined_predictions = torch.argmax(combined_probabilities, dim=-1).tolist()
                        # print('used low confidence')
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
        [{'text': ['Sun is very very very hot.']}, {'text': ['Sun is very very very hot.']}]
    )
    print(results)
