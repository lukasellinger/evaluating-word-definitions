from datasets import Dataset, DatasetDict
from transformers import AutoModel

from config import DB_URL, HF_WRITE_TOKEN
from models.evidence_selection_model import EvidenceSelectionModel

model = AutoModel.from_pretrained('selection_model', trust_remote_code=True, add_pooling_layer=False, safe_serialization=True)
model.push_to_hub("lukasellinger/evidence_selection_model-v1", token=HF_WRITE_TOKEN, private=True)


dataset_query = """
with unique_claims as (
select distinct dd.id, dd.claim, dd.label, dd.evidence_wiki_url, dd.set_type
from def_dataset dd)
select uq.id, uq.claim, uq.label, docs.document_id, docs.text,
       docs.lines, se.evidence_lines as evidence_lines
from unique_claims as uq
    join selected_evidence se on uq.id = se.claim_id
    join documents docs on docs.document_id = uq.evidence_wiki_url
where uq.set_type = '{set_type}'
group by uq.id, docs.document_id
"""

dataset_query = """
select dd.id, dd.claim, dd.label, docs.document_id, docs.text,
       docs.lines, group_concat(dd.evidence_sentence_id) as evidence_lines
from def_dataset dd
    join documents docs on docs.document_id = dd.evidence_wiki_url
where set_type='{set_type}'
group by dd.id, evidence_annotation_id, evidence_wiki_url
"""


train_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='train'), con=DB_URL)
dev_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='dev'), con=DB_URL)
test_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='test'), con=DB_URL)

combined_datasets = DatasetDict({
    "train": train_dataset_raw,
    "dev": dev_dataset_raw,
    "test": test_dataset_raw
})

combined_datasets.push_to_hub("lukasellinger/evidence_selection-v1", private=True, token=HF_WRITE_TOKEN)
