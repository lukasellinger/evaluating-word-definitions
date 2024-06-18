from datasets import Dataset, DatasetDict

from config import DB_URL, HF_WRITE_TOKEN

dataset_query0 = """
with unique_claims as (
select distinct dd.id, dd.claim, dd.short_claim, dd.label, dd.evidence_wiki_url, dd.set_type
from def_dataset dd)
select uq.id, uq.claim as claim, uq.short_claim, uq.label, docs.document_id, docs.text,
       docs.lines, se.evidence_lines as evidence_lines, GROUP_CONCAT(af.fact, '--;--') as atomic_facts
from unique_claims as uq
    join selected_evidence se on uq.id = se.claim_id
    join documents docs on docs.document_id = uq.evidence_wiki_url
    left join atomic_facts_fever_short_dissim af on af.claim_id = uq.id
where uq.set_type = '{set_type}' and 51=51--'{set_type}' and 10=10
group by uq.id, docs.document_id
"""

dataset_query = """
with unique_claims as (
select distinct dd.id, dd.claim, dd.evidence_sentence_id, dd.short_claim, dd.label, dd.evidence_wiki_url, dd.set_type
from def_dataset dd)
select uq.id, uq.claim as claim, uq.short_claim, uq.label, docs.document_id, docs.text,
       docs.lines, GROUP_CONCAT(uq.evidence_sentence_id) as evidence_lines -- GROUP_CONCAT(distinct af.fact, '--;--') as atomic_facts
from unique_claims as uq
    join documents docs on docs.document_id = uq.evidence_wiki_url
    -- left join atomic_facts_fever_short_dissim af on af.claim_id = uq.id
where uq.set_type = '{set_type}' and label != 'NOT ENOUGH INFO'-- ''{set_type}' and 51=51--'{set_type}' and 10=10
group by uq.id, docs.document_id
"""

dataset_query1 = """
select gdd.*, GROUP_CONCAT(af.fact, '--;--') as atomic_facts
from squad_dataset gdd
left join atomic_facts_squad_dissim af on af.claim_id = gdd.id and 1=1
group by gdd.id
"""

#dataset = Dataset.from_sql(dataset_query, con=DB_URL)
#dataset.push_to_hub("lukasellinger/squad_claim_verification_dissim-v1", private=True, token=HF_WRITE_TOKEN)
train_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='train'), cache_dir=None, con=DB_URL)
dev_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='dev'), cache_dir=None, con=DB_URL)
test_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='test'), cache_dir=None, con=DB_URL)

combined_datasets = DatasetDict({
     "train": train_dataset_raw,
     "dev": dev_dataset_raw,
     "test": test_dataset_raw
})

combined_datasets.push_to_hub("lukasellinger/fever_evidence_selection-v1", private=True, token=HF_WRITE_TOKEN)

