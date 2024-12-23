"""Module for Evidence Fetcher."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from fetchers.wikipedia import Wikipedia


class EvidenceFetcher(ABC):
    """
    Abstract base class for fetching evidence related to words.
    """

    def __call__(self, batch: List[Dict], only_intro: bool = True, word_lang: str = 'de'):
        """
        Fetch evidences for a batch of words.

        :param batch: List of dictionaries containing 'word' and 'translated_word'.
        :param only_intro: Flag to fetch only the introduction.
        :param word_lang: Language code for the word.
        :return: Tuple of lists: evidence words and evidence details.
        """
        return self.fetch_evidences_batch(batch, only_intro, word_lang)

    @abstractmethod
    def fetch_evidences(self,
                        word: str | None = None,
                        translated_word: str | None = None,
                        search_word: str | None = None,
                        only_intro: bool = True,
                        word_lang: str = 'de') -> Tuple[str, List[dict]]:
        """
        Fetch evidences for a single word.

        :param word: The word to fetch evidence for.
        :param translated_word: The translated word to use as fallback.
        :param search_word:
        :param only_intro: Flag to fetch only the introduction.
        :param word_lang: Language code for the word.
        :return: Tuple containing the word and its evidence.
        """

    @abstractmethod
    def fetch_evidences_batch(self, batch: List[Dict], only_intro: bool = True,
                              word_lang: str = 'de') -> Tuple[List[str], List[List[dict]]]:
        """
        Fetch evidences for a batch of words.

        :param batch: List of dictionaries containing 'word' and 'translated_word'.
        :param only_intro: Flag to fetch only the introduction.
        :param word_lang: Language code for the word.
        :return: Tuple of lists: evidence words and evidence details.
        """

    @abstractmethod
    def get_max_intro_sent_idx(self) -> Dict:
        """
        Fetch dict keeping the index of the last sentence of every intro passage.
        Right now, only offline needs to be supported.
        """


class WikipediaEvidenceFetcher(EvidenceFetcher):
    """
    EvidenceFetcher implementation that fetches evidence from Wikipedia.
    """

    OFFLINE_WIKI = 'lukasellinger/wiki_dump_2024-09-27'

    def __init__(self,
                 offline: bool = True, source_lang: str = 'en', split_level: str = 'sentence'):
        """
        Initialize the WikipediaEvidenceFetcher.

        :param offline: Whether to use offline Wikipedia data.
        :param source_lang: The source language for Wikipedia data.
        """
        self.offline = offline
        self.split_level = split_level
        self.wiki = Wikipedia(use_dataset=self.OFFLINE_WIKI) if offline else Wikipedia(
            source_lang=source_lang)

    def fetch_evidences(self,
                        word: Optional[str] = None, translated_word: Optional[str] = None,
                        search_word: Optional[str] = None,
                        only_intro: bool = True,
                        word_lang: str = 'de') -> Tuple[str, List[dict]]:
        evid_words, evids = self.fetch_evidences_batch(
            [{'word': word, 'translated_word': translated_word, 'search_word': search_word}],
            only_intro=only_intro, word_lang=word_lang
        )
        return evid_words[0], evids[0]

    def fetch_evidences_batch(self, batch: List[Dict], only_intro: bool = True,
                              word_lang: str = 'de') -> Tuple[List[str], List[List[dict]]]:
        # Validate batch contents based on mode (offline or online)
        required_keys = ['search_word'] if self.offline else ['word', 'translated_word']
        for entry in batch:
            for key in required_keys:
                assert key in entry and entry[
                    key], f'Key "{key}" is missing or has an invalid value in batch entry: {entry}'

        # Fetch evidences for each entry in the batch
        evidence_batch = [
            {
                'word': wiki_word,
                'evidences': [
                    {'title': page, 'line_indices': list(range(len(lines))), 'lines': lines}
                    for page, lines in texts],
            }
            for entry in batch
            for texts, wiki_word in [self.wiki.get_pages(
                word=entry.get('word'),
                fallback_word=entry.get('translated_word'),
                word_lang=word_lang,
                only_intro=only_intro,
                search_word=entry.get('search_word') if self.offline else None,
                split_level=self.split_level
            )]
        ]

        # Unpack evidences and words
        evid_words = [entry['word'] for entry in evidence_batch]
        evids = [entry['evidences'] for entry in evidence_batch]

        return evid_words, evids

    def get_max_intro_sent_idx(self) -> dict:
        return self.wiki.get_offline_max_intro_sent_idx() if self.offline else {}


if __name__ == "__main__":
    fetcher = WikipediaEvidenceFetcher()
    result = fetcher.fetch_evidences_batch([
        {'search_word': 'censorship'},
        {'search_word': 'printed circuit board'}
    ])
    print(result)
