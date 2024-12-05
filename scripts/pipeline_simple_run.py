"""Simple pipeline run script."""
from pipeline_module.claim_splitter import DisSimSplitter
from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.fact_pipeline import Pipeline
from pipeline_module.statement_verifier import ModelStatementVerifier
from pipeline_module.translator import OpusMTTranslator

evid_selector = ModelEvidenceSelector(evidence_selection='mmr')
stm_verifier = ModelStatementVerifier()

pipeline = Pipeline(translator=OpusMTTranslator(),
                    claim_splitter=DisSimSplitter(),
                    evid_fetcher=WikipediaEvidenceFetcher(offline=False),
                    evid_selector=evid_selector,
                    stm_verifier=stm_verifier,
                    lang='en')
print(pipeline.verify(claim='A football is round and Obama is American.', only_intro=True))
