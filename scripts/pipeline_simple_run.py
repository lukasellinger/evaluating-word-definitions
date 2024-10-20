"""Simple pipeline run script."""
from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.pipeline import Pipeline
from pipeline_module.sentence_connector import ColonSentenceConnector
from pipeline_module.statement_verifier import ModelStatementVerifier
from pipeline_module.translator import OpusMTTranslator

evid_selector = ModelEvidenceSelector(evidence_selection='mmr')
stm_verifier = ModelStatementVerifier(
    model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
offline_evid_fetcher = WikipediaEvidenceFetcher(offline=False)

pipeline = Pipeline(translator=OpusMTTranslator(),
                    sent_connector=ColonSentenceConnector(),
                    claim_splitter=None,
                    evid_fetcher=WikipediaEvidenceFetcher(offline=False),
                    evid_selector=evid_selector,
                    stm_verifier=stm_verifier,
                    lang='en')
pipeline.verify(word='unicorn',
                claim='mythical horse with a single horn')
