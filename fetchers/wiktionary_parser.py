from wikitextprocessor import Wtp

from wiktextract import WiktionaryConfig, WiktextractContext, parse_page


class WiktionaryParser:
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

    def get_wiktionary_glosses(self, word, text):
        parsed = parse_page(self.wxr, word, text)
        all_glosses = []
        for entry in parsed:
            senses = entry.get('senses', [])
            for sense in senses:
                glosses = sense.get('glosses')
                if len(glosses) > 1:
                    topics = sense.get('topics', [])
                    tags = sense.get('tags', [])
                    all_glosses.append(f'{entry.get("word")} means: {glosses[1]}')
        return all_glosses
