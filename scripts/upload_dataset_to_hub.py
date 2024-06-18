from datasets import Dataset

from config import DB_URL, HF_WRITE_TOKEN

dataset_query = """
with unique_claims as (
select distinct dd.id, dd.claim, dd.label, dd.evidence_wiki_url, dd.set_type
from def_dataset dd)
select uq.id, uq.claim, uq.label, docs.document_id, docs.text,
       docs.lines, se.evidence_lines as evidence_lines, GROUP_CONCAT(af.fact, '--;--') as atomic_facts
from unique_claims as uq
    join selected_evidence se on uq.id = se.claim_id
    join documents docs on docs.document_id = uq.evidence_wiki_url
    left join atomic_facts af on af.claim_id = uq.id
where uq.set_type = 'dev' --'{set_type}' and 10=10
group by uq.id, docs.document_id
"""

dataset_query = """
select gdd.*, GROUP_CONCAT(af.fact, '--;--') as atomic_facts
from squad_dataset gdd
left join atomic_facts_squad_dissim af on af.claim_id = gdd.id and 1=1
group by gdd.id
"""

dataset = Dataset.from_sql(dataset_query, con=DB_URL)
dataset.push_to_hub("lukasellinger/squad_claim_verification_dissim-v1", private=True, token=HF_WRITE_TOKEN)
# train_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='train'), con=DB_URL)
# dev_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='dev'), con=DB_URL)
# test_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='test'), con=DB_URL)
#
# combined_datasets = DatasetDict({
#     "train": train_dataset_raw,
#     "dev": dev_dataset_raw,
#     "test": test_dataset_raw
# })
#
# combined_datasets.push_to_hub("lukasellinger/claim_verification_atomic-v1", private=True, token=HF_WRITE_TOKEN)

