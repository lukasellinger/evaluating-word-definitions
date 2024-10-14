from copy import deepcopy
from typing import Dict, List, Optional

from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset.def_dataset import Fact, process_lines, split_text
from general_utils.fever_scorer import fever_score
from general_utils.reader import JSONLineReader
from general_utils.utils import process_sentence_wiki, build_fever_instance
from pipeline_module.claim_splitter import ClaimSplitter
from pipeline_module.evidence_fetcher import EvidenceFetcher, WikipediaEvidenceFetcher
from pipeline_module.evidence_selector import EvidenceSelector, ModelEvidenceSelector
from pipeline_module.sentence_connector import SentenceConnector, ColonSentenceConnector
from pipeline_module.statement_verifier import StatementVerifier, ModelEnsembleStatementVerifier
from pipeline_module.translator import Translator


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

        evid_words, evids = self.evid_fetcher(evid_fetcher_input, word_lang=self.lang,
                                              only_intro=only_intro)

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
                     'predicted': -1,
                     'in_wiki': 'No'})
            else:
                filtered_batch.append(entry)
                filtered_evids.append(evid)
                filtered_translations.append({**translation, 'word': word})

        if not filtered_batch:
            return outputs

        processed_batch = self.sent_connector(filtered_translations)

        if self.claim_splitter:
            processed_batch = self.claim_splitter([entry['text'] for entry in processed_batch])

        evids_batch = self.evid_selector(processed_batch, filtered_evids)
        factualities = self.stm_verifier(processed_batch, evids_batch)

        for factuality, evidence, entry in zip(factualities, evids_batch, filtered_batch):
            outputs.append({'word': entry.get('word'),
                            'claim': entry.get('text'),
                            **factuality,
                            'selected_evidences': evidence,
                            'in_wiki': 'Yes'})
        return outputs

    def verify(self, word: str, claim: str, search_word: Optional[str] = None,
               only_intro: bool = True):
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
        return self.verify_batch([entry], only_intro=only_intro)[0]

    def verify_test_select(self, batch: List[Dict], only_intro: bool = True, max_evidence_count: int = 3, top_k: int = 3):
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
                     'predicted': -1}
                )
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
            processed_batch = [{'text': entry.get('english_claim', entry['claim']),
                                'splits': entry[f'{split_type}_facts'].split('--;--')} for entry in filtered_batch]
        else:
            if isinstance(self.sent_connector, ColonSentenceConnector):
                processed_batch = self.sent_connector([{'word': entry['document_search_word'],
                                                        'text': entry.get('english_claim',
                                                                          entry['claim'])} for entry
                                                       in filtered_batch])
            else:
                processed_batch = [{'text': entry['connected_claim']} for entry in filtered_batch]

        evids_batch = self.evid_selector(processed_batch, evids, max_evidence_count, top_k)
        return evids_batch

    def verify_test_batch(self, batch: List[Dict], only_intro: bool = True, max_evidence_count: int = 3, top_k: int = 3):
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
                     'predicted': -1,
                     'in_wiki': 'No'
                     })
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
            processed_batch = [{'text': entry.get('english_claim', entry['claim']),
                                'splits': entry[f'{split_type}_facts'].split('--;--')} for entry in filtered_batch]
        else:
            if isinstance(self.sent_connector, ColonSentenceConnector):
                processed_batch = self.sent_connector([{'word': entry['document_search_word'],
                                                        'text': entry.get('english_claim',
                                                                          entry['claim'])} for entry
                                                       in filtered_batch])
            else:
                processed_batch = [{'text': entry['connected_claim']} for entry in filtered_batch]

        evids_batch = self.evid_selector(processed_batch, evids, max_evidence_count, top_k)
        factualities = self.stm_verifier(processed_batch, evids_batch)

        for factuality, evidence, entry in zip(factualities, evids_batch, filtered_batch):
            outputs.append({'id': entry.get('id'),
                            'word': entry.get('word'),
                            'claim': entry.get('claim'),
                            'connected_claim': entry.get('connected_claim'),
                            'label': entry.get('label'),
                            **factuality,
                            'evidence': self._prep_evidence_output(evidence),
                            'in_wiki': 'Yes'
                            })
        return outputs

    def _prep_evidence_output(self, evidence):
        max_intro_sent_indices = self.evid_fetcher.get_max_intro_sent_idx()
        for entry in evidence:
            entry['in_intro'] = entry.get('line_idx') <= max_intro_sent_indices.get(
                entry.get('title'), -1)
        return evidence

    def verify_test_dataset(self, dataset, batch_size: int = 4, output_file_name: str = '',
                            only_intro: bool = True, max_evidence_count: int = 3, top_k: int = 3):
        """
        Verify a test dataset.

        :param dataset: Dataset to verify.
        """
        dataset = dataset.to_list()

        outputs, gt_labels, pr_labels = [], [], []
        not_in_wiki = 0
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            output = self.verify_test_batch(batch, only_intro=only_intro, max_evidence_count=max_evidence_count, top_k=top_k)

            for entry in output:
                if entry['predicted'] != -1:
                    gt_labels.append(Fact[entry['label']].to_factuality())
                    pr_labels.append(Fact[entry['predicted']].to_factuality())
                else:
                    not_in_wiki += 1

            outputs.extend(output)
        if output_file_name:
            JSONLineReader().write(f'{output_file_name}.jsonl', outputs)

        return outputs, classification_report(gt_labels, pr_labels, zero_division=0,
                                              digits=4), not_in_wiki

    def mark_summary_sents_test_batch(self, batch):
        evids_batch = [{'word': entry.get('word'),
                        'translated_word': entry.get('english_word', entry.get('word')),
                        'search_word': entry['document_search_word']} for entry in batch]

        _, intro_evids_batch = self.evid_fetcher(evids_batch, word_lang=self.lang, only_intro=True)
        max_summary_line_indices = {
            doc['title']: max(map(int, doc['line_indices']))
            for intro_evid in intro_evids_batch
            for doc in intro_evid
        }
        return max_summary_line_indices


class FeverPipeline:
    def __init__(self,
                 claim_splitter: Optional[ClaimSplitter],
                 evid_selector: EvidenceSelector,
                 stm_verifier: StatementVerifier):
        self.claim_splitter = claim_splitter
        self.evid_selector = evid_selector
        self.stm_verifier = stm_verifier

    def get_gold_evids(self, batch):
        filtered_batch = []
        evid_batch = []
        for entry in batch:
            if entry['label'] == 'NOT_ENOUGH_INFO':
                continue
            selected_evid_lines = entry['selected_evidence_lines'].split(',')
            evid = self.prepare_evid(entry)[0]
            evidences = []
            for line in selected_evid_lines:
                sent_idx = evid['line_indices'].index(line)
                evidences.append({'title': evid['title'],
                                  'line_idx': line,
                                  'text': evid['lines'][sent_idx]})
            filtered_batch.append(entry)
            evid_batch.append(evidences)
        return filtered_batch, evid_batch

    def prepare_evid(self, entry):
        lines = process_lines(entry['lines'])
        processed_lines = []
        line_numbers = []
        for line in lines.split('\n'):
            line = process_sentence_wiki(line)
            line_number, text = split_text(line)
            processed_lines.append(text)
            line_numbers.append(line_number)
        return [{'title': entry['document_id'],
                       'line_indices': line_numbers,
                       'lines': processed_lines}]

    def verify_test_batch(self, batch: List[Dict], gold_evidence: bool = False):
        """
        Verify a test batch of claims.

        :param batch: List of dictionaries containing claims.
        :return: Evidences for the batch.
        """
        if gold_evidence:
            batch, evids_batch = self.get_gold_evids(batch)
        else:
            evids = []
            for entry in batch:
                evid = self.prepare_evid(entry)
                evids.append(evid)
            evids_batch = self.evid_selector([{'text': entry['claim']} for entry in batch], evids)

        if self.claim_splitter:
            split_type = type(self.claim_splitter).__name__.split('Splitter')[0]
            processed_batch = [{'text': entry['claim'],
                                'splits': entry[f'{split_type}_facts'].split('--;--')} for entry in batch]
        else:
            processed_batch = [{'text': entry['claim']} for entry in batch]

        factualities = self.stm_verifier(processed_batch, evids_batch)

        outputs, fever_instances = [], []
        for factuality, evidence, entry in zip(factualities, evids_batch, batch):
            outputs.append({'id': entry.get('id'),
                            'word': entry.get('document_id'),
                            'claim': entry.get('claim'),
                            'label': entry.get('label'),
                            **factuality,
                            'evidence': evidence
                            })
            fever_instances.append(build_fever_instance(entry.get('label'),
                                                        entry['evidence_lines'].split(';') if entry['evidence_lines'] else [],
                                                        entry['document_id'],
                                                        factuality.get('predicted'),
                                                        [(line.get('title'), line.get('line_idx'))
                                                         for line in evidence]))
        return outputs, fever_instances

    def verify_test_dataset(self, dataset, batch_size: int = 4, output_file_name: str = '', gold_evidence: bool = False):
        """
        Verify a test dataset.

        :param dataset: Dataset to verify.
        """
        dataset = dataset.to_list()

        outputs, gt_labels, pr_labels, fever_instances = [], [], [], []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            output, fever_instance = self.verify_test_batch(batch, gold_evidence=gold_evidence)

            for entry in output:
                assert entry['predicted'] != -1, 'prediction == -1 can not happen for FeverPipeline'

                gt_labels.append(Fact[entry['label']].to_factuality())
                pr_labels.append(Fact[entry['predicted']].to_factuality())
            fever_instances.extend(fever_instance)
            outputs.extend(output)
        if output_file_name:
            JSONLineReader().write(f'{output_file_name}.jsonl', outputs)

        fever_report = {'strict_score': fever_score(fever_instances)[0],
                        'gold_label': fever_score(fever_instances, use_gold_labels=True)[0]}
        return outputs, classification_report(gt_labels, pr_labels, zero_division=0, digits=4), fever_report


if __name__ == "__main__":
    evid_selector = ModelEvidenceSelector(evidence_selection='mmr')
    stm_verifier = ModelEnsembleStatementVerifier(
        model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
    offline_evid_fetcher = WikipediaEvidenceFetcher()
    pipeline = Pipeline(None, None, None, offline_evid_fetcher, evid_selector,
                         stm_verifier, 'en')
    result = pipeline.verify_test_batch([{'word': 'ERTU',
                                      'document_search_word': 'glacier',
                                          'in_wiki': 'Yes',
                                    'connected_claim': 'A glacier is an ice mass resulting from snow with a clearly defined catchment area, which moves independently due to the slope, structure of the ice, temperature, and the shear stress resulting from the mass of the ice and the other factors.'},
                                     ])
