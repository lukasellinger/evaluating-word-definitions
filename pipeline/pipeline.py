"""Pipelines for the claim verification process."""
from typing import List, Dict, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import BigBirdModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import cosine_similarity

from database.db_retriever import FeverDocDB
from dataset.def_dataset import Fact, process_sentence, process_lines, split_text
from fetchers.wikipedia import Wikipedia
from models.claim_verification_model import ClaimVerificationModel
from models.evidence_selection_model import EvidenceSelectionModel
from utils import rank_docs


class Pipeline:
    """General Pipeline. Implement fetch_evidence, select_evidence, verify_claim."""

    def verify(self, word: str, claim: str) -> Dict:
        """
        Verify a claim related to a word.
        :param word: Word associated to the claim.
        :param claim: Claim to be verified.
        :return: dict containing factuality, atomic claim factualities and selected evidences.
        """
        ev_sents = self.fetch_evidence(word)
        selected_evidences = self.select_evidence(claim, ev_sents)   # we need to know the line and the page the info was taken from
        selected_ev_sents = [evidence[2] for evidence in selected_evidences]
        atomic_claims = self.process_claim(claim)

        total_factuality = 0
        factualities = []
        for atomic_claim in atomic_claims:
            factuality = self.verify_claim(atomic_claim, selected_ev_sents)
            total_factuality += 1 if factuality == Fact.SUPPORTS else 0
            factualities.append(factuality)

        return {'factuality': total_factuality / len(atomic_claims),
                'factualities': factualities,
                'evidences': [(evidence[0], evidence[1]) for evidence in selected_evidences]}

    @staticmethod
    def process_claim(claim: str) -> List[str]:
        """Process a claim. E.g. split it into its atomic facts."""
        return [claim]

    def fetch_evidence(self, word: str) -> List[Tuple[str, List[str], List[str]]]:
        """
        Fetch the information of the word inside the knowledge base.
        :param word: Word, for which we need information.
        :return: List of sentences, representing all information known to the word.
        """

    def select_evidence(self, claim: str, evidence_list: List[Tuple[str, List[str], List[str]]]) -> List[Tuple[str, str, str]]:
        """
        Select sentences possibly containing evidence for the claim.
        :param claim: Claim to be verified.
        :param evidence_list: Sentences to choose from. Can be from multiple sources.
        :return: List of sentences, possibly containing evidence.
        """

    def verify_claim(self, claim: str, sentences: List[str]) -> Fact:
        """
        Verify the claim using sentences as evidence.
        :param claim: Claim to be verified.
        :param sentences: Sentences to use as evidence.
        :return: either Fact.SUPPORTS, Fact.REFUTES or Fact.NOT_ENOUGH_INFO
        """


class ModelPipeline(Pipeline):
    """Pipeline using llm models."""

    def __init__(self, selection_model=None, selection_model_tokenizer=None,
                 verification_model=None, verification_model_tokenizer=None):
        super().__init__()
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

    def _build_selection_model_input(self, claim: str, sentences: List[str]):
        encoded_sequence = []
        sentence_mask = []
        for i, sentence in enumerate(sentences):
            encoded_sentence = self.selection_model_tokenizer.encode(sentence)  # [1:-1]  # + [1]
            encoded_sequence += encoded_sentence
            sentence_mask += [i] * len(encoded_sentence)
            # sentence_mask += [int(line_number)] + [0] * (len(encoded_line) - 1)  # try only with cls token
            # encoded_sequence.append(self.selection_model_tokenizer.sep_token_id)
            # sentence_mask.append(-1)

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
        model_inputs = self._build_verification_model_input(claim, sentences)
        with torch.no_grad():
            output = self.verification_model(**model_inputs)
            predicted = torch.softmax(output['logits'], dim=-1)
            predicted = torch.argmax(predicted, dim=-1).item()
        return Fact(predicted)

    def _build_verification_model_input(self, claim: str, sentences: list[str]):
        hypothesis = ' '.join(sentences)
        model_inputs = self.verification_model_tokenizer(hypothesis, claim)

        return {'input_ids': torch.tensor(model_inputs['input_ids']).unsqueeze(0),
                'attention_mask': torch.tensor(model_inputs['attention_mask']).unsqueeze(0)}


class TestPipeline(ModelPipeline):
    """Pipeline used for test purposes."""

    def fetch_evidence(self, word: str) -> list[tuple[str, list[str], list[str]]]:
        with FeverDocDB() as db:
            lines = db.get_doc_lines(word)

        lines = process_lines(lines)
        processed_lines = []
        line_numbers = []
        for line in lines.split('\n'):
            line = process_sentence(line)
            line_number, text = split_text(line)
            processed_lines.append(text)
            line_numbers.append(line_number)
        return [(word, line_numbers, processed_lines)]

    @staticmethod
    def process_claim(claim: str) -> list[str]:
        with FeverDocDB() as db:
            facts = db.read("""SELECT DISTINCT af.fact
                                         FROM atomic_facts af 
                                         JOIN def_dataset dd ON af.claim_id = dd.id
                                         WHERE dd.claim = ?""", params=(claim,))
        return [fact[0] for fact in facts] if facts else [claim]

    def select_evidence(self, claim: str, evidence_list: list[tuple[str, list[str], list[str]]], top_k=3) -> list[tuple[str, str, str]]:
        page, line_numbers, sentences = evidence_list[0]  # in test case we only have one page

        claim_model_input, sentences_model_input = self._build_selection_model_input(claim,
                                                                                     sentences)
        with torch.no_grad():
            claim_embedding = self.selection_model(**claim_model_input)
            sentence_embeddings = self.selection_model(**sentences_model_input)
            claim_similarities = cosine_similarity(claim_embedding, sentence_embeddings, dim=2)
            top_indices = torch.topk(claim_similarities,
                                     k=min(top_k, claim_similarities.size(1)))[1].squeeze(0)

        return [(page, line_numbers[idx], sentences[idx]) for idx in top_indices]


class WikiPipeline(ModelPipeline):
    """Pipeline using Wikipedia."""

    def __init__(self, selection_model=None, selection_model_tokenizer=None,
                 verification_model=None, verification_model_tokenizer=None):
        super().__init__(selection_model, selection_model_tokenizer, verification_model,
                         verification_model_tokenizer)
        self.wiki = Wikipedia()

    def fetch_evidence(self, word: str) -> list[tuple[str, list[str], list[str]]]:
        summaries = self.wiki.get_summaries(word, k=20)  # TODO line numbers
        return [(page, [str(i) for i in range(len(lines))], lines) for page, lines in summaries]

    def select_evidence(self, claim: str, evidence_list: list[tuple[str, list[str], list[str]]], top_k=3,
                        max_evidence_count=3) -> list[tuple[str, str, str]]:
        if len(evidence_list) > max_evidence_count:
            ranked_indices = rank_docs(claim, [" ".join(entry[2]) for entry in evidence_list],
                                       k=max_evidence_count)
            evidence_list = [evidence_list[i] for i in ranked_indices]

        sentence_similarities = []
        for page, line_numbers, sentences in evidence_list:
            claim_model_input, sentences_model_input = self._build_selection_model_input(claim,
                                                                                         sentences)
            with torch.no_grad():
                claim_embedding = self.selection_model(**claim_model_input)
                sentence_embeddings = self.selection_model(**sentences_model_input)
                claim_similarities = cosine_similarity(claim_embedding,
                                                       sentence_embeddings, dim=2).tolist()[0]
                sentence_similarity = [(page, *values) for values in zip(line_numbers, sentences, claim_similarities)]
                sentence_similarities.extend(sentence_similarity)

        sorted_sentences = sorted(sentence_similarities, key=lambda x: x[3], reverse=True)
        return [(sentence[0], sentence[1], sentence[2]) for sentence in sorted_sentences[:top_k]]


if __name__ == "__main__":
    pipeline = WikiPipeline()

    tests = [('Albania', 'Albania is a member of NATO.', 1),
             ('Spain', 'Spain is in Europe.', 1),
             ('Reds_-LRB-film-RRB-', 'Reds is an epic drama film.', 1),
             ('Unpredictable_-LRB-Jamie_Foxx_album-RRB-', 'Unpredictable was an album.', 1),
             ('Ruth_Negga', 'Ruth Negga is a film actress.', 1),
             ('Inspectah_Deck', 'Inspectah Deck is stateless.', 0),
             ('Drake_-LRB-musician-RRB-', 'Drake is only German.', 0),
             ('Overwatch_-LRB-video_game-RRB-', 'Overwatch is a board game.', 0),
             ('Ad-Rock', 'Ad-Rock is single.', 0),
             ('Gujarat', 'Gujarat is in Western Boston.', 0)]

    tests1 = [('Albania', 'Albania is a member of NATO.', -1),
             # is not in summary text for current wiki
             ('Spain', 'Spain is in Europe.', 1),
             ('Reds', 'Reds is an epic drama film.', 1),
             ('Unpredictable', 'Unpredictable was an album.', 1),
             ('Ruth Negga', 'Ruth Negga is a film actress.', 1),
             ('Inspectah Deck', 'Inspectah Deck is stateless.', 0),  # ??
             ('Drake', 'Drake is only German.', 0),
             ('Overwatch', 'Overwatch is a board game.', 0),
             ('Ad-Rock', 'Ad-Rock is single.', -1),  # is not in summary text for current wiki
             ('Gujarat', 'Gujarat is in Western Boston.', 0)]

    pr_labels = []
    gt_labels = []
    fever_instances = []
    for word_test, claim_test, gt_label in tqdm(tests1, desc='Verifying claim'):
        factuality_test = pipeline.verify(word=word_test, claim=claim_test)

        gt_labels += [gt_label] * len(factuality_test)
        pr_labels.extend([fact.to_factuality() for fact in factuality_test])

    acc = accuracy_score(gt_labels, pr_labels)
    f1_weighted = f1_score(gt_labels, pr_labels, average='weighted')
    f1_macro = f1_score(gt_labels, pr_labels, average='macro')

    print(acc)
    print(f1_weighted)
    print(f1_macro)

    ######
    # test pipeline
    # 0.9
    # 0.94
    # 0.63
    # wiki pipeline
    # 0.9
    # 0.89
    # 0.85
    ######
