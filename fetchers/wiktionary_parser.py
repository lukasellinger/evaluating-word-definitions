"""Module for parsing Wiktionary."""
from typing import List

from wikitextprocessor import Wtp
from wiktextract import WiktextractContext, WiktionaryConfig, parse_page


class WiktionaryParser:
    """Parser of Wiktionary Pages."""
    def __init__(self):
        config = WiktionaryConfig(
            capture_language_codes=['de', 'en'],
            capture_translations=True,
            capture_pronunciation=True,
            capture_linkages=True,
            capture_compounds=True,
            capture_redirects=True,
            capture_examples=True,
            capture_etymologies=True,
            capture_inflections=True,
        )
        self.wxr = WiktextractContext(Wtp(), config)

    def get_wiktionary_glosses(self, word: str, text: str) -> List[str]:
        """
        Extracts and returns glosses (meanings) from a Wiktionary page for a given word.

        :param word: The word to extract glosses for.
        :param text: The raw Wiktionary text for the word.
        :return: A list of glosses (meanings) extracted from the page in the format
        'word means: gloss'.
        """
        parsed = parse_page(self.wxr, word, text)
        all_glosses = []
        for entry in parsed:
            senses = entry.get('senses', [])
            for sense in senses:
                glosses = sense.get('glosses')
                if len(glosses) > 1:
                    # topics = sense.get('topics', [])
                    # tags = sense.get('tags', [])
                    all_glosses.append(f'{entry.get("word")} means: {glosses[1]}')
        return all_glosses
