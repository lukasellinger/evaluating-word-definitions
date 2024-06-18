"""Pipelines for the claim verification process."""
from typing import List, Dict, Tuple

import torch
from transformers import BigBirdModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import cosine_similarity

from database.db_retriever import FeverDocDB
from dataset.def_dataset import Fact, process_sentence, process_lines, split_text
from general_utils.fact_extractor import FactExtractor
from fetchers.wikipedia import Wikipedia
from models.claim_verification_model import ClaimVerificationModel
from models.evidence_selection_model import EvidenceSelectionModel
from general_utils.utils import rank_docs


class Pipeline:
    """General Pipeline. Implement fetch_evidence, select_evidence, verify_claim."""

    def __init__(self, word_lang: str = None):
        self.word_lang = word_lang
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def verify(self, word: str, claim: str, fallback_word: str = None, split_facts: bool = True, only_intro: bool = True, atomic_claims = None) -> Dict:
        """
        Verify a claim related to a word.
        :param word: Word associated to the claim.
        :param claim: Claim to be verified.
        :return: dict containing factuality, atomic claim factualities and selected evidences.
        """
        output = {'factuality': -1,
                  'factualities': [],
                  'evidences': []}
        ev_sents, wiki_word = self.fetch_evidence(word, fallback_word, only_intro)

        if ev_sents:
            claim = f"{wiki_word}: {claim}"
            selected_evidences = self.select_evidence(claim, ev_sents)   # we need to know the line and the page the info was taken from
            selected_ev_sents = [evidence[2] for evidence in selected_evidences]

            if atomic_claims:  # in order to use already computed atomic claims
                atomic_claims = [f'{wiki_word}: {atomic_claim}' for atomic_claim in atomic_claims]
            else:
                atomic_claims = self.process_claim(claim, split_facts=split_facts)
                if split_facts:  # otherwise wiki_word twice
                    atomic_claims = [f'{wiki_word}: {atomic_claim}' for atomic_claim in atomic_claims]

            total_factuality = 0
            factualities = []
            for atomic_claim in atomic_claims:
                factuality = self.verify_claim(atomic_claim, selected_ev_sents)
                total_factuality += 1 if factuality == Fact.SUPPORTED else 0
                factualities.append((atomic_claim, factuality))

            output['factuality'] = total_factuality / len(atomic_claims)
            output['factualities'] = factualities
            output['evidences'] = selected_evidences
        return output

    @staticmethod
    def process_claim(claim: str, split_facts: bool = True) -> List[str]:
        """Process a claim. E.g. split it into its atomic facts."""
        return [claim]

    def fetch_evidence(self, word: str, fallback_word: str = None, only_intro: bool = True) -> List[Tuple[str, List[str], List[str]]]:
        """
        Fetch the information of the word inside the knowledge base.
        :param word: Word, for which we need information.
        :param fallback_word:
        :param only_intro: Whether to only get text of the intro section. Default: True.
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
                 verification_model=None, verification_model_tokenizer=None, word_lang=None):
        super().__init__(word_lang)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not selection_model:
            model_name = 'google/bigbird-roberta-large'
            model = BigBirdModel.from_pretrained(model_name)
            selection_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
            selection_model = EvidenceSelectionModel(model).to(self.device)
        self.selection_model = selection_model
        self.selection_model_tokenizer = selection_model_tokenizer
        self.selection_model.eval()

        if not verification_model:
            model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
            verification_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            verification_model = ClaimVerificationModel(model).to(self.device)
        self.verification_model = verification_model
        self.verification_model_tokenizer = verification_model_tokenizer
        self.verification_model.eval()

    def _build_selection_model_input(self, claim: str, sentences: List[str]):
        encoded_sequence = []
        sentence_mask = []
        for i, sentence in enumerate(sentences):
            encoded_sentence = self.selection_model_tokenizer.encode(sentence)[1:-1]  # + [1]
            encoded_sequence += encoded_sentence
            sentence_mask += [i] * len(encoded_sentence)
            # sentence_mask += [int(i)] + [-1] * (len(encoded_sentence) - 1)  # try only with cls token
            encoded_sequence.append(self.selection_model_tokenizer.sep_token_id)
            sentence_mask.append(-1)

        unique_sentence_numbers = set(sentence_mask)
        sentence_masks = []
        for num in unique_sentence_numbers:
            if num == -1:
                continue
            sentence_masks.append([1 if val == num else 0 for val in sentence_mask])

        return (self.selection_model_tokenizer(claim, return_tensors='pt').to(self.device),
                {'input_ids': torch.tensor(encoded_sequence).unsqueeze(0).to(self.device),
                 'attention_mask': torch.ones(len(encoded_sequence)).unsqueeze(0).to(self.device),
                 'sentence_mask': torch.tensor(sentence_masks).unsqueeze(0).to(self.device)})

    def verify_claim(self, claim: str, sentences: list[str]) -> Fact:
        model_inputs = self._build_verification_model_input(claim, sentences)
        with torch.no_grad():
            logits = self.verification_model(**model_inputs)['logits']
            predicted = torch.softmax(logits, dim=-1)
            # predicted = torch.argmax(predicted, dim=-1).item()
            predicted[:, 1] += predicted[:, 2]
            predicted = predicted[:, :2]
            predicted = torch.argmax(predicted, dim=-1).item()
        return Fact(predicted)

    def _build_verification_model_input(self, claim: str, sentences: list[str]):
        hypothesis = ' '.join(sentences)
        model_inputs = self.verification_model_tokenizer(hypothesis, claim)

        return {'input_ids': torch.tensor(model_inputs['input_ids']).unsqueeze(0).to(self.device),
                'attention_mask': torch.tensor(model_inputs['attention_mask']).unsqueeze(0).to(self.device)}


class TestPipeline(ModelPipeline):
    """Pipeline used for test purposes."""

    def fetch_evidence(self, word: str, fallback_word: str = None, only_intro: bool = True) -> list[tuple[str, list[str], list[str]]]:
        with FeverDocDB() as db:
            lines = db.get_doc_lines(word)

        if not lines:
            return [], fallback_word

        lines = process_lines(lines)
        processed_lines = []
        line_numbers = []
        for line in lines.split('\n'):
            line = process_sentence(line)
            line_number, text = split_text(line)
            processed_lines.append(text)
            line_numbers.append(line_number)
        return [(word, line_numbers, processed_lines)], fallback_word

    @staticmethod
    def process_claim(claim: str,  split_facts: bool = True) -> list[str]:
        if split_facts:
            with FeverDocDB() as db:
                facts = db.read("""SELECT DISTINCT af.fact
                                            FROM atomic_facts af
                                            JOIN def_dataset dd ON af.claim_id = dd.id
                                            WHERE dd.claim = ?""", params=(claim,))
            return [fact[0] for fact in facts] if facts else [claim]
        return [claim]

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
                 verification_model=None, verification_model_tokenizer=None, word_lang=None):
        super().__init__(selection_model, selection_model_tokenizer, verification_model,
                         verification_model_tokenizer, word_lang)
        self.wiki = Wikipedia()
        self.fact_extractor = FactExtractor()

    def process_claim(self, claim: str, split_facts: bool = True) -> list[str]:
        if split_facts:
            facts = self.fact_extractor.get_atomic_facts(claim)
            return facts.get('facts') if facts.get('facts') else [claim]
        return [claim]

    def fetch_evidence(self, word: str, fallback_word: str = None, only_intro: bool = True) -> list[tuple[str, list[str], list[str]]]:
        texts, wiki_word = self.wiki.get_pages(word, fallback_word, self.word_lang, only_intro=only_intro)
        #texts = self.wiki.get_texts(word, k=20, only_intro=only_intro)  # TODO line numbers
        return [(page, [str(i) for i in range(len(lines))], lines) for page, lines in texts], wiki_word

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

    def mark_summary_sents(self, word):
        ev_sentences_short = self.fetch_evidence(word, only_intro=True)
        max_summary_line_numbers = [(doc[0], max(int(line_number) for line_number in doc[1])) for doc in ev_sentences_short]
        return dict(max_summary_line_numbers)
