"""Module for making api call to wikipedia."""
import re
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from requests import Response

from spacy_utils import split_into_sentences


class Wikipedia:
    """Wrapper for wikipedia api calls."""

    USER_AGENT = 'summaryBot (lu.ellinger@gmx.de)'
    URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, user_agent: str = None):
        if user_agent:
            self.USER_AGENT = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})

    def _get_response(self, params) -> Response:
        return self.session.get(url=self.URL, params=params)

    def get_texts(self, word: str, k: int = 20, only_intro: bool = False) -> List[Tuple[str, List[str]]]:
        """
        Get the summary of the top k most similar (according to wikipedia search) articles of a word
        on wikipedia.
        :param word: Word to get the summaries of.
        :param k: amount of summaries to get.
        :param only_intro: Whether to only get text of the intro section. Default: True.
        :return: List of summaries split in their sentences.
        """
        page_ids = self.get_top_k_search_results(word, k)
        return self.get_text_from_page_ids(page_ids, only_intro)

    def get_top_k_search_results(self, search_txt: str, k: int = 10) -> List[int]:
        """
        Get the top k search results for a search text on wikipedia.
        :param search_txt: txt to search for.
        :param k: amount of search results to get.
        :return: list of page_ids.
        """
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srlimit": k,
            "srsearch": search_txt
        }

        response = self._get_response(params)
        data = response.json()

        return [entry.get('pageid') for entry in data.get('query', {}).get('search', [])]

    def get_text_from_page_ids(self, page_ids: List[int], only_intro: bool = False) -> List[Tuple[str, List[str]]]:
        """
        Get the summary of the pages of page_ids on wikipedia.
        :param page_ids: Page_ids to get the summaries of.
        :param only_intro: Whether to only get text of the intro section. Default: True.
        :return: List of summaries split in their sentences.
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

        texts = []
        while True:
            response = self._get_response(params)
            data = response.json()
            for _, value in data.get('query', {}).get('pages', {}).items():
                if value.get('extract') and value.get('title'):
                    title = str(value.get('title'))
                    text = str(value.get('extract'))
                    # section headlines disturb sentence splitting
                    text = re.sub(r'(==+)\s*[^=]+?\s*==+', '.', text)
                    sentences = split_into_sentences(text)
                    texts.append((title, sentences))   # TODO line numbers
            if continue_batch := data.get('continue'):
                params.update(continue_batch)
            else:
                break
        return texts


if __name__ == "__main__":
    wiki = Wikipedia()
    print(wiki.get_texts("Hitler", k=20))
