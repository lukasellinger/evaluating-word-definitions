"""Module for making api call to wikipedia."""
import re
from typing import List, Tuple, Dict

import pandas as pd
import requests
from datasets import load_dataset
from requests import Response
from transformers import RobertaTokenizer

from fetchers.wiktionary_parser import WiktionaryParser
from general_utils.spacy_utils import split_into_sentences, split_into_passage_sentences
from general_utils.utils import (generate_case_combinations, split_into_passages,
                                 remove_duplicate_values)


class Wikipedia:
    """Wrapper for wikipedia api calls."""

    USER_AGENT = 'summaryBot (lu.ellinger@gmx.de)'
    BASE_URL = "https://{source_lang}.{site}.org/w/api.php"

    def __init__(self, source_lang: str = 'en', user_agent: str = None, use_dataset: str = ''):
        self.USER_AGENT = user_agent or self.USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})
        self.base_url = self.BASE_URL.format(source_lang=source_lang, site='{site}')
        self.offline_backend = None
        if use_dataset:
            self.offline_backend = self._prepare_offline_backend(use_dataset)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.wiktionary_parser = WiktionaryParser()

    def _prepare_offline_backend(self, dataset) -> Dict:
        backend_dataset = pd.DataFrame(load_dataset(dataset).get('train'))
        offline_backend = backend_dataset.groupby('search_word').apply(
            lambda x: x.to_dict(orient='records')).to_dict()
        return offline_backend

    def get_offline_max_intro_sent_idx(self) -> Dict:
        """
        Retrieves the maximum introductory sentence index for each title in the offline backend.

        :return: A dictionary mapping titles to their maximum introductory sentence index.
        """
        return {e['title']: int(e['intro_end_sent_idx']) for k, v in self.offline_backend.items()
                for e in v}

    def _get_response(self, params, site: str, source_lang=None) -> Response:
        assert site in {'wikipedia', 'wiktionary'}
        url = self.base_url.format(site=site) if not source_lang else self.BASE_URL.format(
            source_lang=source_lang,
            site=site)
        return self.session.get(url=url, params=params)

    def get_texts(self,
                  word: str, k: int = 20,
                  only_intro: bool = True, site: str = 'wikipedia') -> List[Tuple[str, List[str]]]:
        """
        Retrieves the summary of the top k most relevant articles for a given word from Wikipedia.

        :param word: The word to search for on Wikipedia.
        :param k: The number of articles to retrieve. Defaults to 20.
        :param only_intro: Whether to retrieve only the introductory section of the articles.
        Defaults to True.
        :param site: The site to query (e.g., 'wikipedia'). Defaults to 'wikipedia'.
        :return: A list of tuples, where each tuple contains the article title and a list of
        sentences.
        """
        page_ids = self.get_top_k_search_results(word, k)
        return self.get_text_from_page_ids(page_ids, only_intro, site=site)

    def get_top_k_search_results(self, search_txt: str, k: int = 10) -> List[int]:
        """
        Retrieves the top k search results for a given search text from Wikipedia.

        :param search_txt: The text to search for on Wikipedia.
        :param k: The number of search results to retrieve. Defaults to 10.
        :return: A list of page IDs corresponding to the search results.
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srlimit": k,
            "srsearch": search_txt
        }

        response = self._get_response(params, site='wikipedia')
        data = response.json()
        return [entry.get('pageid') for entry in data.get('query', {}).get('search', [])]

    def get_text_from_page_ids(self,
                               page_ids: List[int], only_intro: bool = True,site: str = 'wikipedia',
                               split_level='sentence',
                               return_raw=False) -> List[Tuple[str, List[str]]]:
        """
        Retrieves text for the given page IDs from Wikipedia.

        :param page_ids: A list of page IDs to retrieve text from.
        :param only_intro: Whether to retrieve only the introductory section of the articles.
        Defaults to True.
        :param site: The site to query (e.g., 'wikipedia'). Defaults to 'wikipedia'.
        :param split_level: The level at which to split the text (e.g., 'sentence').
        Defaults to 'sentence'.
        :param return_raw: Whether to return the raw text without processing. Defaults to False.
        :return: A list of tuples containing the article title and a list of sentences.
        """
        if not page_ids:
            return [('', [])]

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": True,
            "pageids": "|".join(map(str, page_ids))
        }

        if only_intro:
            params['exintro'] = "true"

        return list(self._fetch_batch(params, site=site, split_level=split_level,
                                      return_raw=return_raw).items())

    def _fetch_batch(self, params: Dict, site: str, sentence_limit: int = 250,
                     split_level: str = 'sentence', return_raw: bool = False) -> Dict:
        """
        Fetch a batch of text data from the specified site with optional cleaning and splitting.

        :param params: Parameters for the API request.
        :param site: Site from which to fetch data ('wikipedia' or 'wiktionary').
        :param sentence_limit: Maximum number of sentences to include if split by sentences.
        :param split_level: Level at which to split the text ('passage', 'sentence', 'none').
        :param return_raw: Whether to return the raw text without cleaning and splitting.
        :return: Dictionary of fetched texts with keys indicating the title and part.
        """
        texts = {}  # dict to get rid of possible duplicates
        while True:
            response = self._get_response(params, site=site)
            data = response.json()

            for page in data.get('query', {}).get('pages', {}).values():
                title, text = str(page.get('title')), page.get('extract', '')

                if title and text:
                    if return_raw:
                        texts.update(self._split_text(title, site, text, split_level='none'))
                    else:
                        if site != 'wiktionary':
                            text = self._clean_text(text)
                        texts.update(
                            self._split_text(title, site, text, split_level, sentence_limit))
            if 'continue' not in data:
                break
            params.update(data['continue'])

        return texts

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r'(==+)\s*[^=]+?\s*==+', '.', text)
        text = re.sub(r'[^\S ]', '.', text)
        text = re.sub(r"\.{2,}", ". ", text)
        text = re.sub(r"^[. ]+", "", text)
        return text

    def _split_text(self, title: str, site: str, text: str, split_level: str = 'sentence',
                    sentence_limit=250):
        """
        Split text based on the specified split-level.

        :param title: Title of the page.
        :param site: Site from which the text is fetched.
        :param text: The text to be split.
        :param split_level: Level at which to split the text ('passage', 'sentence', 'none').
        :param sentence_limit: Maximum number of sentences to include if split by sentences.
        :return: Dictionary of split texts with keys indicating the title and part.
        """
        if split_level not in {'passage', 'sentence', 'passage_sentences', 'none'}:
            raise ValueError(
                "split_level needs to be 'passage', 'sentence', 'passage_sentences', or 'none'")

        texts = {}
        key_base = f'{title} ({site})' if site else title

        if split_level == 'none':
            texts[key_base] = text
        elif key_base.endswith('(wiktionary)'):
            word = key_base.split(' (wik')[0]
            sentences = self.wiktionary_parser.get_wiktionary_glosses(word, text)
            if sentences:
                texts[key_base] = sentences[:sentence_limit]
        elif split_level == 'passage':
            passages = split_into_passages(split_into_sentences(text), self.tokenizer)
            texts = {f'{key_base} {i}': passage for i, passage in enumerate(passages)}
        elif split_level == 'passage_sentences':
            passages = split_into_passage_sentences(text)
            texts = {f'{key_base} {i}': passage for i, passage in enumerate(passages)}
        elif split_level == 'sentence':
            sentences = split_into_sentences(text)
            texts[key_base] = sentences[:sentence_limit]

        return texts

    def get_text_from_title(self, page_titles: List[str], site: str = 'wikipedia',
                            only_intro: bool = True, split_level='sentence',
                            return_raw=False) -> Dict:
        """
        Retrieves text content from Wikipedia based on the provided page titles.

        :param page_titles: List of page titles to retrieve.
        :param site: The site to query (default is 'wikipedia').
        :param only_intro: Whether to retrieve only the introductory section (default: True).
        :param split_level: The level at which to split the text (
        'passage', 'sentence', 'passage_sentences', 'none')
        :param return_raw: Whether to return raw text without cleaning or splitting
        (default: False).
        :return: A dictionary with page titles as keys and the corresponding fetched text as values.
        """
        results = {}
        for batch_pages in self._chunk(page_titles, 50):  # wikipedia api supports a maximum of 50
            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "explaintext": True,
                "titles": "|".join(batch_pages),
                "redirects": True  # Follow redirects, e.g. Light bulb to Electric light
            }
            if only_intro:
                params['exintro'] = "true"
            results.update(
                self._fetch_batch(params, site, split_level=split_level, return_raw=return_raw))
        return results

    def find_similar_titles(self, search_term, k: int = 1000) -> List[str]:
        """
        Finds and returns titles similar to the given search term using Wikipedia's search
        functionality.

        :param search_term: The term to search for.
        :param k: The number of similar titles to return (default: 1000).
        :return: A list of similar page titles.
        """
        params = {
            "action": "opensearch",
            "format": "json",
            "search": f"{search_term}_(",  # senses are disambiguated with (), e.g. run (song)
            "limit": k
        }
        response = self._get_response(params, site='wikipedia')
        data = response.json()

        similar_titles = [search_term] + [
            entry for entry in data[1]
            if re.fullmatch(fr'{search_term}(?: \(.+\))?', entry, flags=re.IGNORECASE)
        ]
        return list(set(similar_titles))

    def get_pages(self, word: str, fallback_word: str = None, word_lang: str = None,
                  only_intro=True,
                  split_level='sentence', return_raw=False, search_word='') -> Tuple[List, any]:
        """
        Retrieves pages related to a word from either online or offline sources, based on the
        backend setup.

        :param word: The word to retrieve pages for.
        :param fallback_word: A fallback translation word in case the original word is not found
        (default: None).
        :param word_lang: The language of the word (default: None).
        :param only_intro: Whether to retrieve only the introductory section of the pages
        (default: True).
        :param split_level: The level at which to split the text
        ('passage', 'sentence', 'passage_sentences', 'none').
        :param return_raw: Whether to return raw text without cleaning or splitting
        (default: False).
        :param search_word: The search term to look for in the offline backend (default: '').
        :return: A tuples, containing the List of pages and the word referenced.
        """
        if self.offline_backend and search_word:
            return self.get_pages_offline(search_word, only_intro, return_raw, split_level)
        else:
            return self.get_pages_online(word, fallback_word, word_lang, only_intro, split_level,
                                         return_raw)

    def get_pages_offline(self, search_word: str, only_intro: bool, return_raw: bool,
                          split_level: str) -> Tuple[List, any]:
        """
        Retrieves pages from the offline backend for a given search term.

        :param search_word: The search term to look for in the offline dataset.
        :param only_intro: Whether to retrieve only the introductory section (default: True).
        :param return_raw: Whether to return raw text without cleaning or splitting
        (default: False).
        :param split_level: The level at which to split the text ('sentence', 'passage', 'none').
        :return: A list of tuples containing page titles and corresponding content.
        """
        entries = self.offline_backend.get(search_word, [])
        texts = {}
        for entry in entries:
            title = entry.get('title')
            text = entry.get('raw_intro_text') if only_intro else entry.get('raw_full_text')
            if not text:  # some wikipages do not have intro sections
                continue
            if return_raw:
                texts.update(self._split_text(title, '', text, split_level='none'))
            else:
                # text = self._clean_text(text)
                texts.update(self._split_text(title, '', text, split_level))
        return list(texts.items()), search_word

    def get_pages_online(self, word: str, fallback_word: str = None, word_lang: str = None,
                         only_intro=True,
                         split_level='sentence', return_raw=False) -> Tuple[List, any]:
        """
        Retrieves pages from Wikipedia online for the given word and language.

        :param word: The word to retrieve pages for.
        :param fallback_word: A fallback word in case the original word could not be translated
        (default: None).
        :param word_lang: The language of the word (default: None).
        :param only_intro: Whether to retrieve only the introductory section (default: True).
        :param split_level: The level at which to split the text ('sentence', 'passage', 'none').
        :param return_raw: Whether to return raw text without cleaning or splitting (default: False).
        :return: A list of tuples containing page titles and corresponding content.
        """
        word = word.lower()  # lower to find all results
        # check word in original language in english dictionary, need full page here
        case_words = generate_case_combinations(word)  # wiktionary titles are case-sensitive
        dict_text_word = self.get_text_from_title(case_words,
                                                  only_intro=False,
                                                  site='wiktionary',
                                                  split_level=split_level,
                                                  return_raw=return_raw)
        pages = dict_text_word

        if word_lang != 'en':
            word = self.translate_word(word, fallback_word, word_lang).lower()
            assert word, "Word could not be translated and no fallback word provided."

            # check translated word in english dictionary, need full page here
            dict_text_translated = self.get_text_from_title([word],
                                                            only_intro=False,
                                                            site='wiktionary',
                                                            split_level=split_level,
                                                            return_raw=return_raw)
            pages.update(dict_text_translated)

        # check normal wikipedia
        if similar_titles := self.find_similar_titles(word):
            wiki_texts = self.get_text_from_title(similar_titles,
                                                  only_intro=only_intro,
                                                  split_level=split_level,
                                                  return_raw=return_raw)
            pages.update(wiki_texts)
        pages = remove_duplicate_values(pages)  # just to be sure no duplicate effort is made.
        return list(pages.items()), word

    def translate_word(self, word: str, fallback_word: str = '', word_lang: str = 'de') -> str:
        """
        Translates a word into another language using Wikipedia's interlanguage links.

        :param word: The word to translate.
        :param fallback_word: A fallback word if no translation is found (default: '').
        :param word_lang: The source language of the word (default: 'de').
        :return: The translated word, or the fallback word if no translation is available.
        """
        interlang_word = self.get_interlanguage_title(word, source_lang=word_lang)
        return interlang_word or fallback_word

    def get_interlanguage_title(self, title, site: str = 'wikipedia', source_lang='de',
                                target_lang="en") -> str | None:
        """
        Retrieves the title of an article in a different language using Wikipedia's interlanguage
        links.

        :param title: The title of the article in the source language.
        :param site: The site to query (default: 'wikipedia').
        :param source_lang: The source language of the article (default: 'de').
        :param target_lang: The target language for translation (default: 'en').
        :return: The translated title, or None if no translation is found.
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "langlinks",
            "lllang": target_lang
        }
        response = self._get_response(params, site, source_lang)
        data = response.json()

        page = next(iter(data.get('query', {}).get('pages', {}).values()), {})
        if langlinks := page.get('langlinks'):
            return langlinks[0].get('*').split(' (')[0]  # there is only one langlink
        return None

    @staticmethod
    def _chunk(items: List[str], size: int) -> List[List[str]]:
        for i in range(0, len(items), size):
            yield items[i:i + size]


if __name__ == "__main__":
    wiki = Wikipedia(use_dataset='lukasellinger/wiki_dump_2024-08-14')

    # full_docs, _ = wiki.get_pages('Hammer', 'Hammer', 'de', only_intro=False, return_raw=True)
    # intro_docs, document_search_word = wiki.get_pages('Hammer', 'Hammer', word_lang='de', only_intro=True, return_raw=True)
    # assert len(full_docs) == len(intro_docs), f'For Hammer, len(intro) != len(full)'

    # wiki.get_text_from_title(['Love (Masaki Suda album)'])
    a = wiki.get_pages_online(word='Apfel', fallback_word='apple', split_level='sentence', only_intro=True,
                               return_raw=False)
