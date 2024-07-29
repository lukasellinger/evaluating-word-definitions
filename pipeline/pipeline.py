from copy import deepcopy
from typing import Dict, List, Optional

from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset.def_dataset import Fact
from general_utils.reader import JSONLineReader
from pipeline.claim_splitter import ClaimSplitter, DisSimSplitter
from pipeline.evidence_fetcher import EvidenceFetcher, WikipediaEvidenceFetcher
from pipeline.evidence_selector import EvidenceSelector, ModelEvidenceSelector
from pipeline.sentence_connector import SentenceConnector, ColonSentenceConnector
from pipeline.statement_verifier import StatementVerifier, ModelStatementVerifier
from pipeline.translator import Translator, OpusMTTranslator


class Pipeline:
    """General Pipeline for fetching evidence, selecting evidence, and verifying claims."""

    def __init__(self,
                 translator: Optional[Translator],
                 sent_connector: SentenceConnector,
                 claim_splitter: Optional[ClaimSplitter],
                 evid_fetcher: EvidenceFetcher,
                 evid_selector: EvidenceSelector,
                 stm_verifier: StatementVerifier,
                 lang: str):
        self.translator = translator
        self.sent_connector = sent_connector
        self.claim_splitter = claim_splitter
        self.evid_fetcher = evid_fetcher
        self.evid_selector = evid_selector
        self.stm_verifier = stm_verifier
        self.lang = lang

    def verify_batch(self, batch: List[Dict], only_intro: bool = True):
        """
        Verify a batch of claims.

        :param batch: List of dictionaries containing 'word' and 'text'.
        :return: List of outputs with factuality and selected evidences.
        """
        processed_batch = deepcopy(batch)

        if self.lang != 'en':
            translation_batch = self.translator(processed_batch)
            evid_fetcher_input = [{**b, 'translated_word': t.get('word')} for b, t in
                                  zip(batch, translation_batch)]
        else:
            translation_batch = processed_batch
            evid_fetcher_input = [{**b, 'translated_word': b.get('word')} for b in batch]

        evid_words, evids = self.evid_fetcher(evid_fetcher_input, word_lang=self.lang, only_intro=only_intro)

        outputs = []
        filtered_batch = []
        filtered_evids = []
        filtered_translations = []
        for entry, evid, word, translation in zip(processed_batch, evids, evid_words,
                                                  translation_batch):
            if not evid:
                outputs.append(
                    {'word': entry.get('word'),
                     'claim': entry.get('text'),
                     'predicted': -1})
            else:
                filtered_batch.append(entry)
                filtered_evids.append(evid)
                filtered_translations.append({**translation, 'word': word})

        if not filtered_batch:
            return outputs

        processed_batch = self.sent_connector(filtered_translations)

        if self.claim_splitter:
            processed_batch = self.claim_splitter(processed_batch)

        evids_batch = self.evid_selector(processed_batch, filtered_evids)
        factualities = self.stm_verifier(processed_batch, evids_batch)

        for factuality, evidence, entry in zip(factualities, evids_batch, filtered_batch):
            outputs.append({'word': entry.get('word'),
                            'claim': entry.get('text'),
                            **factuality,
                            'selected_evidences': evidence})
        return outputs

    def verify(self, word: str, claim: str, search_word: Optional[str] = None, only_intro: bool = True):
        """
        Verify a single claim.

        :param word: The word to verify.
        :param claim: The claim to verify.
        :param search_word: Optional search word for evidence fetching.
        :return: Verification result.
        """
        entry = {'word': word, 'text': claim}
        if search_word:
            entry['search_word'] = search_word
        return self.verify_batch([entry], only_intro=only_intro)

    def verify_test_batch(self, batch: List[Dict], only_intro: bool = True):
        """
        Verify a test batch of claims.

        :param batch: List of dictionaries containing claims.
        :return: Evidences for the batch.
        """
        filtered_batch = []
        outputs = []
        for entry in batch:
            if entry['in_wiki'] == 'No':
                outputs.append(
                    {'id': entry.get('id'),
                     'word': entry.get('word'),
                     'claim': entry.get('claim'),
                     'connected_claim': entry.get('connected_claim'),
                     'label': entry.get('label'),
                     'predicted': -1})
            else:
                filtered_batch.append(entry)

        evid_fetcher_input = [{'word': entry['word'],
                               'translated_word': entry.get('english_word', entry['word']),
                               'search_word': entry['document_search_word']} for entry in
                              filtered_batch]

        _, evids = self.evid_fetcher(evid_fetcher_input, word_lang=self.lang, only_intro=only_intro)

        if not filtered_batch:
            return outputs

        if self.claim_splitter:
            split_type = type(self.claim_splitter).__name__.split('Splitter')[0]
            processed_batch = [{'splits': entry[f'{split_type}_facts']} for entry in filtered_batch]
        else:
            if isinstance(self.sent_connector, ColonSentenceConnector):
                processed_batch = self.sent_connector(filtered_batch)
            else:
                processed_batch = [{'text': entry['connected_claim']} for entry in filtered_batch]

        evids_batch = self.evid_selector(processed_batch, evids)
        factualities = self.stm_verifier(processed_batch, evids_batch)

        for factuality, evidence, entry in zip(factualities, evids_batch, filtered_batch):
            outputs.append({'id': entry.get('id'),
                            'word': entry.get('word'),
                            'claim': entry.get('claim'),
                            'connected_claim': entry.get('connected_claim'),
                            'label': entry.get('label'),
                            **factuality,
                            'evidence': evidence
                            })
        return outputs

    def _prep_evidence_output(self, evidence):
        max_intro_sent_indices = self.evid_fetcher.mark_summary_sents_test_batch()
        for entry in evidence:
            entry['in_intro'] = entry.get('line_idx') <= max_intro_sent_indices.get(entry.get('title'), -1)
        return evidence

    def verify_test_dataset(self, dataset, batch_size: int = 4, output_file: str = '', only_intro: bool = True):
        """
        Verify a test dataset.

        :param dataset: Dataset to verify.
        """
        dataset = dataset.to_list()

        outputs, gt_labels, pr_labels = [], [], []
        not_in_wiki = 0
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            output = self.verify_test_batch(batch, only_intro=only_intro)

            for entry in output:
                if entry['predicted'] != -1:
                    gt_labels.append(Fact[entry['label']].to_factuality())
                    pr_labels.append(Fact[entry['predicted']].to_factuality())
                else:
                    not_in_wiki += 1

            outputs.extend(output)
        if output_file:
            JSONLineReader().write(output_file, outputs)

        return outputs, classification_report(gt_labels, pr_labels, zero_division=0,
                                              digits=4), not_in_wiki

    def mark_summary_sents_test_batch(self, batch):
        evids_batch = [{'word': entry.get('word'),
                        'translated_word': entry.get('english_word', entry.get('word')),
                        'search_word': entry['document_search_word']} for entry in batch]

        _, intro_evids_batch = self.evid_fetcher(evids_batch, word_lang=self.lang, only_intro=True)
        max_summary_line_numbers = {
            doc[0]: max(map(int, doc[1]))
            for intro_evid in intro_evids_batch
            for doc in intro_evid
        }
        return max_summary_line_numbers


if __name__ == "__main__":
    translator = OpusMTTranslator()
    sent_connector = ColonSentenceConnector()
    claim_splitter = None  # DisSimSplitter()
    evid_fetcher = WikipediaEvidenceFetcher()
    evid_selector = ModelEvidenceSelector()
    stm_verifier = ModelStatementVerifier(
        model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
    lang = 'de'

    pipeline = Pipeline(translator, sent_connector, claim_splitter, evid_fetcher, evid_selector,
                        stm_verifier, lang)
    #result = pipeline.verify_batch([ {'word': 'ERTU', 'text': 'die staatliche Rundfunkgesellschaft Ägyptens', 'search_word': 'ertu'},
    #                                 {'word': 'Kindergewerkschaft', 'text': 'I like to swim and dance', 'search_word': "children's union"}])
    #print(result)
    #result = pipeline.verify_batch([{'word': 'ERTU', 'text': 'die staatliche Rundfunkgesellschaft Ägyptens', 'search_word': 'ertu'}])
    #print(result)
    dataset = load_dataset('lukasellinger/german_dpr_claim_verification_dissim-v1', split='train')
    pipeline.verify_test_dataset(dataset, 4)
