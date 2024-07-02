"""Module for making api call to wikipedia."""
import re
from typing import List, Tuple, Dict
import requests
from requests import Response

from general_utils.spacy_utils import split_into_sentences
from general_utils.utils import generate_case_combinations, split_into_passages


class Wikipedia:
    """Wrapper for wikipedia api calls."""

    USER_AGENT = 'summaryBot (lu.ellinger@gmx.de)'
    URL = "https://{source_lang}.{site}.org/w/api.php"

    def __init__(self, source_lang: str = 'en', user_agent: str = None):
        if user_agent:
            self.USER_AGENT = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})
        self.URL = self.URL.format(source_lang=source_lang, site='{site}')

    def _get_response(self, params, site: str, source_lang=None) -> Response:
        assert site in {'wikipedia', 'wiktionary'}

        if source_lang:
            return self.session.get(url=Wikipedia.URL.format(source_lang=source_lang, site=site), params=params)
        return self.session.get(url=self.URL.format(site=site), params=params)

    def get_texts(self, word: str, k: int = 20, only_intro: bool = True, site: str = 'wikipedia') -> List[Tuple[str, List[str]]]:
        """
        Get the summary of the top k most similar (according to wikipedia search) articles of a word
        on wikipedia.
        :param word: Word to get the summaries of.
        :param k: amount of summaries to get.
        :param only_intro: Whether to only get text of the intro section. Default: True.
        :return: List of summaries split in their sentences.
        """
        page_ids = self.get_top_k_search_results(word, k)
        return self.get_text_from_page_ids(page_ids, only_intro, site=site)

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

        response = self._get_response(params, site='wikipedia')
        data = response.json()

        return [entry.get('pageid') for entry in data.get('query', {}).get('search', [])]

    def get_text_from_page_ids(self, page_ids: List[int], only_intro: bool = True, site: str = 'wikipedia', split_text=False) -> List[Tuple[str, List[str]]]:
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

        return list(self._fetch_batch(params, site=site, split_text=split_text).items())

    def _fetch_batch(self, params: Dict, site: str, sentence_limit: int = 250, split_text=False) -> Dict:
        texts = {}  # dict to get rid of possible duplicates
        while True:
            response = self._get_response(params, site=site)
            data = response.json()
            for _, value in data.get('query', {}).get('pages', {}).items():
                if value.get('extract') and value.get('title'):
                    title = str(value.get('title'))
                    text = str(value.get('extract'))
                    # section headlines disturb sentence splitting
                    text = re.sub(r'(==+)\s*[^=]+?\s*==+', '.', text)

                    # convert all whitespace chars except ' ' to '.'
                    text = re.sub(r'[^\S ]', '.', text)
                    text = re.sub(r"\.{2,}", ". ", text)
                    text = re.sub(r"^[. ]+", "", text)

                    if split_text:
                        passages = split_into_passages(text)
                        texts = {f'{title} ({site}) {i}': passage for i, passage in enumerate(passages)}  # TODO line numbers
                    else:
                        sentences = split_into_sentences(text)
                        texts[f'{title} ({site})'] = sentences[:sentence_limit]  # TODO line numbers
            if continue_batch := data.get('continue'):
                params.update(continue_batch)
            else:
                break
        return texts

    def get_text_from_title(self, page_titles: List[str], site: str = 'wikipedia', only_intro: bool = True, split_text=False) -> Dict:
        def process():
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

            return self._fetch_batch(params, site, split_text=split_text)

        results = {}
        for i in range(0, len(page_titles), 50):  # wikipedia api supports a maximum of 50
            batch_pages = page_titles[i:i + 50]
            results.update(process())
        return results

    def find_similar_titles(self, search_term, k: int = 1000) -> List[str]:
        params = {
            "action": "opensearch",
            "format": "json",
            "search": search_term + "_(",  # senses are disambiguated with (), e.g. run (song)
            "limit": k
        }

        response = self._get_response(params, site='wikipedia')
        data = response.json()

        similar_titles = [search_term]
        for entry in data[1]:  # page titles
            if re.fullmatch(f'{search_term}(?: \(.+\))?', entry, flags=re.IGNORECASE):
                similar_titles.append(entry)

        return list(set(s.lower() for s in similar_titles))

    def get_pages(self, word: str, fallback_word: str = None, word_lang: str = None, only_intro=True, split_text=False):
        word = word.lower()  # lower to find all results

        # check word in original language in english dictionary, need full page here
        case_words = generate_case_combinations(word)  # wiktionary titles are case-sensitive
        dict_text_word = self.get_text_from_title(case_words,
                                                  only_intro=False,
                                                  site='wiktionary',
                                                  split_text=split_text)
        pages = dict_text_word

        if word_lang != 'en':
            word = self.translate_word(word, fallback_word, word_lang)
            assert word, "Word could not be translated and no fallback word provided."
            word = word.lower()

        # check translated word in english dictionary, need full page here
        dict_text_translated = self.get_text_from_title([word],
                                                        only_intro=False,
                                                        site='wiktionary',
                                                        split_text=split_text)
        pages.update(dict_text_translated)

        # check normal wikipedia
        if similar_titles := self.find_similar_titles(word):
            wiki_texts = self.get_text_from_title(similar_titles,
                                                  only_intro=only_intro,
                                                  split_text=split_text)
            pages.update(wiki_texts)

        return list(pages.items()), word

    def translate_word(self, word: str, fallback_word=None, word_lang: str = 'de') -> str:
        if interlang_word := self.get_interlanguage_title(word, source_lang=word_lang):
            return interlang_word
        else:
            return fallback_word

    def get_interlanguage_title(self, title, site: str = 'wikipedia', source_lang='de', target_lang="en"):
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "langlinks",
            "lllang": target_lang
        }

        response = self._get_response(params, site, source_lang)
        data = response.json()
        page = next(iter(data.get('query', {}).get('pages', {}).values()))
        if langlinks := page.get('langlinks'):
            word = langlinks[0].get('*')  # there is only one langlink
            return word.split(' (')[0]


if __name__ == "__main__":
    wiki = Wikipedia()
    wiki.get_pages('Chaos', 'Chaos', only_intro=True, word_lang='de')
    a = wiki.find_similar_titles('Apple')
    print('hi')
    #print(wiki.get_pages('a', 'data a', word_lang='de', only_intro=True))