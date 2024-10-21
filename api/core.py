from datasets import load_dataset

from pydantic_models import Example
from pipeline import ProgressPipeline
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.statement_verifier import ModelStatementVerifier
from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.sentence_connector import ColonSentenceConnector
from pipeline_module.translator import OpusMTTranslator

datasets = [
    {
        "id": 1,
        "name": "German DPR",
        "lang": "de",
        "examples": [
            Example(word=example['word'], definition=example['claim'])
            for example in load_dataset('lukasellinger/german_dpr-claim_verification', split='test')
        ]
    },
    {
        "id": 2,
        "name": "Wiktionary Mini",
        "lang": "de",
        "examples": [
            Example(word=example['word'], definition=example['claim'])
            for example in
            load_dataset('lukasellinger/german_wiktionary-claim_verification-mini', split='test')
        ]
    },
    {
        "id": 3,
        "name": "SQuAD",
        "lang": "en",
        "examples": [
            Example(word=example['word'], definition=example['claim'])
            for example in load_dataset('lukasellinger/squad-claim_verification', split='test')
        ]
    },
    {
        "id": 4,
        "name": "SHROOM",
        "lang": "en",
        "examples": [
            Example(word=example['word'], definition=example['claim'])
            for example in load_dataset('lukasellinger/shroom-claim_verification', split='test')
        ]
    },
    # {
    #     "id": 5,
    #     "name": "Wiktionary Large",
    #     "lang": "de",
    #     "examples": [
    #         Example(word=example['word'], definition=example['claim'])
    #         for example in load_dataset('lukasellinger/german_wiktionary-claim_verification-large', split='test')
    #     ]
    # }
]

# Initialize the pipeline instance
pipeline = ProgressPipeline(
    OpusMTTranslator(),
    ColonSentenceConnector(),
    None,
    WikipediaEvidenceFetcher(offline=False),
    ModelEvidenceSelector(),
    ModelStatementVerifier(),
    'de'
)
