import asyncio
from copy import deepcopy

from general_utils.spacy_utils import get_main_entity
from pipeline_module.claim_splitter import ClaimSplitter
from pipeline_module.evidence_fetcher import EvidenceFetcher
from pipeline_module.evidence_selector import EvidenceSelector
from pipeline_module.statement_verifier import StatementVerifier
from pipeline_module.translator import Translator


class Pipeline:
    """General Pipeline for fetching evidence, selecting evidence, and verifying claims."""

    def __init__(self,
                 translator: Translator | None,
                 claim_splitter: ClaimSplitter | None,
                 evid_fetcher: EvidenceFetcher,
                 evid_selector: EvidenceSelector,
                 stm_verifier: StatementVerifier,
                 lang: str):
        self.translator = translator
        self.claim_splitter = claim_splitter
        self.evid_fetcher = evid_fetcher
        self.evid_selector = evid_selector
        self.stm_verifier = stm_verifier
        self.lang = lang

    def verify_batch(self, batch: list[dict], only_intro: bool = True) -> list[dict]:
        """
        Verify a batch of claims by fetching, selecting, and verifying evidence.

        :param batch: list of dictionaries containing 'text'.
        :param only_intro: Flag to indicate if only the introductory section of documents should
        be considered.
        :return: list of outputs with factuality and selected evidences.
        """
        processed_batch = deepcopy(batch)

        if self.lang != 'en':
            translation_batch = self.translator.translate_claim_batch(processed_batch)
        else:
            translation_batch = processed_batch

        if self.claim_splitter:
            processed_batch = self.claim_splitter([entry['text'] for entry in translation_batch])
        else:
            processed_batch = [{**entry, 'splits': [entry['text']]} for entry in translation_batch]

        entity_batch = []
        for entry in processed_batch:
            entities = []
            for split in entry['splits']:
                entities.append(get_main_entity(split))
            entity_batch.append({**entry, 'words': entities})

        evids_words_batch, evids_batch = [], []
        for entry in entity_batch:
            if all(entry['words']):
                evid_fetcher_input = [{'word': word, 'translated_word': word} for word in
                                      entry['words']]
                evid_words, evids = self.evid_fetcher(evid_fetcher_input, word_lang=self.lang,
                                                      only_intro=only_intro)
            else:
                evid_words, evids = [], []
            evids_words_batch.append(evid_words)
            evids_batch.append(evids)

        outputs = []
        filtered_batch = []
        filtered_evids = []
        for entry, evids, words in zip(processed_batch, evids_batch, evids_words_batch):
            if not all(evids) or not evids:
                outputs.append(
                    {'claim': entry.get('text'),
                     'predicted': -1,
                     'in_wiki': 'No'})
            else:
                filtered_batch.append(entry)
                filtered_evids.append(evids)

        if not filtered_batch:
            return outputs

        factualities = []
        for entry, evid in zip(filtered_batch, filtered_evids):
            selected_evids = self.evid_selector([{'text': split} for split in entry['splits']],
                                                evid)
            factuality = self.stm_verifier.verify_splitted_claim(entry, selected_evids)
            factualities.append(factuality)

        for factuality, entry in zip(factualities, filtered_batch):
            outputs.append({'claim': entry.get('text'),
                            **factuality,
                            'in_wiki': 'Yes'})
        return outputs

    def verify(self, claim: str, only_intro: bool = True) -> dict:
        """
        Verify a single claim.

        :param claim: The claim to verify.
        :param only_intro: Flag to indicate if only the introductory section of documents should
        be considered.
        :return: Verification result.
        """
        entry = {'text': claim}
        return self.verify_batch([entry], only_intro=only_intro)[0]


class ProgressPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_callback = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    async def verify(self, claim: str, only_intro: bool = True):
        if self.progress_callback:
            await self.progress_callback("startingVerification")

        if self.translator and self.lang != 'en':
            if self.progress_callback:
                await self.progress_callback("translating")
            translated_claim = await asyncio.to_thread(self.translator.translate_text, claim)
        else:
            translated_claim = claim

        if self.claim_splitter:
            if self.progress_callback:
                await self.progress_callback("splittingClaim")
            splitted_entry = await asyncio.to_thread(self.claim_splitter.get_atomic_claims,
                                                     translated_claim)
        else:
            splitted_entry = {'text': translated_claim, 'splits': [translated_claim]}

        if self.progress_callback:
            await self.progress_callback("extractingEntities")
        splitted_entry['words'] = []
        for split in splitted_entry['splits']:
            splitted_entry['words'].append(get_main_entity(split))

        if self.progress_callback:
            await self.progress_callback("fetchingEvidence")

        if all(splitted_entry['words']):
            evid_fetcher_input = [{'word': word, 'translated_word': word} for word in
                                  splitted_entry['words']]
            evid_words, evids = self.evid_fetcher(evid_fetcher_input, word_lang=self.lang,
                                                  only_intro=only_intro)
        else:
            evid_words, evids = [], []

        if not evids or not all(evids):
            if self.progress_callback:
                await self.progress_callback("noEvidenceFound")
            return {'claim': claim, 'predicted': '', 'in_wiki': 'No'}

        if self.progress_callback:
            await self.progress_callback("selectingEvidence")

        selected_evids = await asyncio.to_thread(self.evid_selector, [{'text': split} for split in
                                                                      splitted_entry['splits']],
                                                 evids)

        if self.progress_callback:
            await self.progress_callback("verifyingStatement")
        factuality = await asyncio.to_thread(self.stm_verifier.verify_splitted_claim, splitted_entry, selected_evids)

        if self.progress_callback:
            await self.progress_callback("verificationComplete")

        return {
            'claim': claim,
            **factuality,
            'in_wiki': 'Yes'
        }
