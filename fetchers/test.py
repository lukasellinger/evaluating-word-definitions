from transformers import AutoTokenizer, AutoModel

from models.evidence_selection_model import EvidenceSelectionModel
from pipeline.pipeline import WikiPipeline

selection_model_tokenizer = AutoTokenizer.from_pretrained(
    'Snowflake/snowflake-arctic-embed-m-long')
model = AutoModel.from_pretrained('lukasellinger/evidence_selection_model-v1',
                                  trust_remote_code=True, add_pooling_layer=False,
                                  safe_serialization=True)
selection_model = EvidenceSelectionModel(model)

# still using base
verification_model = None
verification_model_tokenizer = None

wiki_pipeline = WikiPipeline(selection_model=selection_model,
                             selection_model_tokenizer=selection_model_tokenizer, word_lang='de')

#print(wiki_pipeline.verify('Light bulb', 'Light bulb: an artificial light source', split_facts=False))
print(wiki_pipeline.verify('Freundschaftsspiel', 'Friendship Game: a sporting match pairing that does not flow into an official rating', 'Friendship game', split_facts=False))