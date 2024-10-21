import asyncio
from typing import Optional

from pipeline_module.pipeline import Pipeline


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

        if self.translator:
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
        evid_words, evids = await asyncio.to_thread(
            self.evid_fetcher,
            [{'word': word, 'translated_word': translated_word, 'search_word': search_word}],
            word_lang=self.lang,
            only_intro=only_intro
        )

        if not evids:
            if self.progress_callback:
                await self.progress_callback("No evidence found")
            return {'word': word, 'claim': claim, 'predicted': -1, 'in_wiki': 'No'}

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