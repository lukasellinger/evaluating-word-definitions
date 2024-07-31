from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

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
    def fetch_evidences(self, word: Optional[str] = None, translated_word: Optional[str] = None, search_word: Optional[str] = None, only_intro: bool = True,
                        word_lang: str = 'de') -> Tuple[
        str, List[Tuple[str, List[str], List[str]]]]:
        """
        Fetch evidences for a single word.

        :param word: The word to fetch evidence for.
        :param translated_word: The translated word to use as fallback.
        :param only_intro: Flag to fetch only the introduction.
        :param word_lang: Language code for the word.
        :return: Tuple containing the word and its evidence.
        """
        pass

    @abstractmethod
    def fetch_evidences_batch(self, batch: List[Dict], only_intro: bool = True,
                              word_lang: str = 'de'):
        """
        Fetch evidences for a batch of words.

        :param batch: List of dictionaries containing 'word' and 'translated_word'.
        :param only_intro: Flag to fetch only the introduction.
        :param word_lang: Language code for the word.
        :return: Tuple of lists: evidence words and evidence details.
        """
        pass

    @abstractmethod
    def get_max_intro_sent_idx(self) -> Dict:
        """
        Fetch dict keeping the index of the last sentence of every intro passage.
        Right now, only offline needs to be supported.
        """
        pass


class WikipediaEvidenceFetcher(EvidenceFetcher):
    """
    EvidenceFetcher implementation that fetches evidence from Wikipedia.
    """

    OFFLINE_WIKI = 'lukasellinger/wiki_dump_2024-07-08'

    def __init__(self, offline: bool = True, source_lang: str = 'en'):
        """
        Initialize the WikipediaEvidenceFetcher.

        :param offline: Whether to use offline Wikipedia data.
        :param source_lang: The source language for Wikipedia data.
        """
        self.offline = offline
        self.wiki = Wikipedia(use_dataset=self.OFFLINE_WIKI) if offline else Wikipedia(
            source_lang=source_lang)

    def fetch_evidences(self, word: Optional[str] = None, translated_word: Optional[str] = None, search_word: Optional[str] = None, only_intro: bool = True,
                        word_lang: str = 'de') -> Tuple[
        str, List[Tuple[str, List[str], List[str]]]]:
        """
        Fetch evidences for a single word using the batch method.

        :param word: The word to fetch evidence for.
        :param translated_word: The translated word to use as fallback.
        :param only_intro: Flag to fetch only the introduction.
        :param word_lang: Language code for the word.
        :return: Tuple containing the word and its evidence.
        """

        evid_words, evids = self.fetch_evidences_batch(
            [{'word': word, 'translated_word': translated_word, 'search_word': search_word}],
            only_intro=only_intro, word_lang=word_lang
        )
        return evid_words[0], evids[0]

    def fetch_evidences_batch(self, batch: List[Dict], only_intro: bool = True,
                              word_lang: str = 'de', offline: bool = True) -> Tuple[
        List[str], List[List[Tuple[str, List[str], List[str]]]]]:
        """
        Fetch evidences for a batch of words.

        :param batch: List of dictionaries containing 'word' and 'translated_word'.
        :param only_intro: Flag to fetch only the introduction.
        :param word_lang: Language code for the word.
        :return: Tuple of lists: evidence words and evidence details.
        """
        # Validate batch contents based on mode (offline or online)
        required_keys = ['search_word'] if offline else ['word', 'translated_word']
        for key in required_keys:
            assert all(key in entry for entry in batch), f'Key {key} missing in batch entries'

        # Fetch evidences for each entry in the batch
        evidence_batch = [
            {
                'word': wiki_word,
                'evidences': [{'title': page, 'line_indices': [i for i in range(len(lines))], 'lines': lines} for page, lines in texts],
            }
            for entry in batch
            for texts, wiki_word in [self.wiki.get_pages(
                word=entry.get('word'),
                fallback_word=entry.get('translated_word'),
                word_lang=word_lang,
                only_intro=only_intro,
                search_word=entry.get('search_word') if self.offline else None
            )]
        ]

        # Unpack evidences and words
        evid_words = [entry['word'] for entry in evidence_batch]
        evids = [entry['evidences'] for entry in evidence_batch]

        return evid_words, evids

    def get_max_intro_sent_idx(self):
        return self.wiki.get_offline_max_intro_sent_idx() if self.offline else {}



if __name__ == "__main__":
    fetcher = WikipediaEvidenceFetcher()
    fetcher.wiki.get_offline_max_intro_sent_idx()
    result = fetcher.fetch_evidences_batch([
        {'search_word': 'censorship'},
        {'search_word': 'printed circuit board'}
    ])
    print(result)
