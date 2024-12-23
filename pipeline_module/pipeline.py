"""Module for Pipelines."""
import asyncio
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import classification_report
from tqdm import tqdm

from dataset.def_dataset import Fact, process_lines, split_text
from general_utils.fever_scorer import fever_score
from general_utils.reader import JSONLineReader
from general_utils.utils import build_fever_instance, process_sentence_wiki
from pipeline_module.claim_splitter import ClaimSplitter
from pipeline_module.evidence_fetcher import (EvidenceFetcher,
                                              WikipediaEvidenceFetcher)
from pipeline_module.evidence_selector import (EvidenceSelector,
                                               ModelEvidenceSelector)
from pipeline_module.sentence_connector import (ColonSentenceConnector,
                                                SentenceConnector)
from pipeline_module.statement_verifier import (ModelStatementVerifier,
                                                StatementVerifier)
from pipeline_module.translator import OpusMTTranslator, Translator


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

    def verify_batch(self, batch: List[Dict], only_intro: bool = True) -> List[Dict]:
        """
        Verify a batch of claims by fetching, selecting, and verifying evidence.

        :param batch: List of dictionaries containing 'word' and 'text'.
        :param only_intro: Flag to indicate if only the introductory section of documents should
        be considered.
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
               only_intro: bool = True) -> Dict:
        """
        Verify a single claim.

        :param word: The word to verify.
        :param claim: The claim to verify.
        :param search_word: Optional search word for evidence fetching.
        :param only_intro: Flag to indicate if only the introductory section of documents should
        be considered.
        :return: Verification result.
        """
        entry = {'word': word, 'text': claim}
        if search_word:
            entry['search_word'] = search_word
        return self.verify_batch([entry], only_intro=only_intro)[0]

    @staticmethod
    def filter_batch_for_wikipedia(batch: List[Dict],
                                   evids_batch: List[List[Dict]],
                                   outputs) -> Tuple[List[Dict], List[List[Dict]], List[Dict]]:
        filtered_batch, filtered_evids = [], []
        for evid in evids_batch:
            evid[:] = [d for d in evid if d.get('title', '').endswith('(wikipedia)')]

        for entry, evid in zip(batch, evids_batch):
            if len(evid) > 0:
                filtered_batch.append(entry)
                filtered_evids.append(evid)
            else:
                outputs.append(
                    {'id': entry.get('id'),
                     'word': entry.get('word'),
                     'claim': entry.get('claim'),
                     'connected_claim': entry.get('connected_claim'),
                     'label': entry.get('label'),
                     'predicted': -1,
                     'in_wiki': 'No'
                     })

        return filtered_batch, filtered_evids, outputs

    def verify_test_batch(self, batch: List[Dict], only_intro: bool = True,
                          max_evidence_count: int = 3, top_k: int = 3,
                          only_wikipedia: bool = False) -> List[Dict]:
        """
        Verify a test batch of claims by fetching, selecting, and verifying evidence.

        :param batch: List of dictionaries containing claims.
        :param only_intro: Flag to indicate if only the introductory section of documents should
        be considered.
        :param max_evidence_count: Maximum number of evidences to consider for each claim.
        :param top_k: Number of top sentences to select for each claim.
        :param only_wikipedia: Whether evidence should only be from wikipedia. Else also wiktionary.
        :return: List of verified claims with evidence.
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

        if only_wikipedia:
            filtered_batch, evids, outputs = self.filter_batch_for_wikipedia(filtered_batch,
                                                                             evids,
                                                                             outputs)

        if not filtered_batch:
            return outputs

        if self.claim_splitter:
            split_type = type(self.claim_splitter).__name__.split('Splitter')[0]
            processed_batch = [{'text': entry.get('english_claim', entry['claim']),
                                'splits': entry[f'{split_type}_facts'].split('--;--')} for entry in
                               filtered_batch]
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

    def _prep_evidence_output(self, evidence: List[Dict]) -> List[Dict]:
        max_intro_sent_indices = self.evid_fetcher.get_max_intro_sent_idx()
        for entry in evidence:
            entry['in_intro'] = entry.get('line_idx') <= max_intro_sent_indices.get(
                entry.get('title'), -1)
        return evidence

    def verify_test_dataset(self,
                            dataset,
                            batch_size: int = 4,
                            output_file_name: str = '',
                            only_intro: bool = True,
                            max_evidence_count: int = 3,
                            top_k: int = 3,
                            only_wikipedia: bool = False) -> Tuple[List[Dict], str, int]:
        """
        Verify a test dataset in batches.

        :param dataset: Dataset to verify.
        :param batch_size: Number of claims to verify at a time.
        :param output_file_name: Optional name of the output file to save results.
        :param only_intro: Flag to indicate if only the introductory section of documents should
        be considered.
        :param max_evidence_count: Maximum number of evidences to consider for each claim.
        :param top_k: Number of top sentences to select for each claim.
        :param only_wikipedia: Whether evidence should only be from wikipedia. Else also wiktionary.
        :return: Tuple containing verification results, classification report, and count of claims
        not found in wiki.
        """
        dataset = dataset.to_list()

        outputs, gt_labels, pr_labels = [], [], []
        not_in_wiki = 0
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]
            output = self.verify_test_batch(batch, only_intro=only_intro,
                                            max_evidence_count=max_evidence_count,
                                            top_k=top_k,
                                            only_wikipedia=only_wikipedia)

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

    def mark_summary_sents_test_batch(self, batch: List[Dict]) -> Dict:
        """
        Mark the summary sentences for a test batch of claims.

        :param batch: List of dictionaries containing claims.
        :return: Dictionary mapping document titles to maximum summary sentence indices.
        """
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
    """Pipeline specifically designed for FEVER (Fact Extraction and Verification) dataset."""

    def __init__(self,
                 claim_splitter: Optional[ClaimSplitter],
                 evid_selector: EvidenceSelector,
                 stm_verifier: StatementVerifier):
        self.claim_splitter = claim_splitter
        self.evid_selector = evid_selector
        self.stm_verifier = stm_verifier

    def get_gold_evids(self, batch: List[Dict]) -> Tuple[List[Dict], List[List[Dict]]]:
        """
        Retrieve the gold-standard evidence for a batch of claims.

        :param batch: List of dictionaries containing claims.
        :return: Tuple containing the filtered batch and corresponding evidence batch.
        """
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

    @staticmethod
    def prepare_evid(entry: Dict) -> List[Dict]:
        """
        Prepare evidence by processing lines from the entry and splitting them.

        :param entry: Dictionary containing evidence lines.
        :return: List of processed evidence.
        """
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

    def verify_test_batch(self,
                          batch: List[Dict],
                          gold_evidence: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Verify a test batch of claims, optionally using gold-standard evidence.

        :param batch: List of dictionaries containing claims.
        :param gold_evidence: Flag to indicate if gold-standard evidence should be used.
        :return: Tuple containing verification results and FEVER instances.
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
                                'splits': entry[f'{split_type}_facts'].split('--;--')} for entry in
                               batch]
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
                                                        entry['evidence_lines'].split(';') if entry[
                                                            'evidence_lines'] else [],
                                                        entry['document_id'],
                                                        factuality.get('predicted'),
                                                        [(line.get('title'), line.get('line_idx'))
                                                         for line in evidence]))
        return outputs, fever_instances

    def verify_test_dataset(self, dataset, batch_size: int = 4, output_file_name: str = '',
                            gold_evidence: bool = False) -> Tuple[List[Dict], str, Dict]:
        """
        Verify a test dataset in batches for the FEVER task.

        :param dataset: Dataset to verify.
        :param batch_size: Number of claims to verify at a time.
        :param output_file_name: Optional name of the output file to save results.
        :param gold_evidence: Flag to indicate if gold-standard evidence should be used.
        :return: Tuple containing verification results, classification report, and
        FEVER score report.
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
        return outputs, classification_report(gt_labels, pr_labels, zero_division=0,
                                              digits=4), fever_report


class ProgressPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    async def verify(self, word: str, claim: str, search_word: Optional[str] = None,
                     only_intro: bool = True):
        if self.progress_callback:
            await self.progress_callback("Starting verification process")

        if self.translator and self.lang != 'en':
            if self.progress_callback:
                await self.progress_callback("Translating input")
            translated = await asyncio.to_thread(self.translator, [{'word': word, 'text': claim}])
            translated = translated[0]
            translated_word = translated.get('word', word)
            translated_claim = translated.get('text', claim)
        else:
            translated_word = word
            translated_claim = claim

        if self.progress_callback:
            await self.progress_callback("Fetching evidence")
        evid_words, evids = self.evid_fetcher(
            [{'word': word, 'translated_word': translated_word, 'search_word': search_word}],
            word_lang=self.lang,
            only_intro=only_intro
        )

        if not evids or all(not sublist for sublist in evids):
            if self.progress_callback:
                await self.progress_callback("No evidence found")
            return {'word': word, 'claim': claim, 'predicted': '', 'in_wiki': 'No'}

        if self.progress_callback:
            await self.progress_callback("Processing claim")
        processed_claim = await asyncio.to_thread(self.sent_connector, [
            {'word': evid_words[0], 'text': translated_claim}])
        processed_claim = processed_claim[0]

        if self.claim_splitter:
            if self.progress_callback:
                await self.progress_callback("Splitting claim")
            processed_claim = await asyncio.to_thread(self.claim_splitter,
                                                      [processed_claim['text']])
            processed_claim = processed_claim[0]

        if self.progress_callback:
            await self.progress_callback("Selecting evidence")
        selected_evids = await asyncio.to_thread(self.evid_selector, [processed_claim], evids)
        selected_evids = selected_evids[0]

        if self.progress_callback:
            await self.progress_callback("Verifying statement")
        factuality = await asyncio.to_thread(self.stm_verifier, [processed_claim], [selected_evids])
        factuality = factuality[0]

        if self.progress_callback:
            await self.progress_callback("Verification complete")

        return {
            'word': word,
            'claim': claim,
            **factuality,
            'selected_evidences': selected_evids,
            'in_wiki': 'Yes'
        }


if __name__ == "__main__":
    pipeline = Pipeline(OpusMTTranslator(),
                        ColonSentenceConnector(),
                        None,
                        WikipediaEvidenceFetcher(),
                        ModelEvidenceSelector(evidence_selection='mmr'),
                        ModelStatementVerifier(
                            model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'),
                        'de')
    result = pipeline.verify_test_batch([{'word': 'ERTU',
                                          'document_search_word': 'glacier',
                                          'in_wiki': 'Yes',
                                          'connected_claim': 'A glacier is an ice mass resulting '
                                                             'from snow with a clearly defined '
                                                             'catchment area, which moves '
                                                             'independently due to the slope, '
                                                             'structure of the ice, temperature, '
                                                             'and the shear stress resulting from '
                                                             'the mass of the ice and the other '
                                                             'factors.'},
                                         ], only_wikipedia=True)
    # pipeline.verify_batch([{'word': 'Apfel', 'text': 'Huhn'}])
