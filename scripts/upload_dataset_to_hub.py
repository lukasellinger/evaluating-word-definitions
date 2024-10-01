from datasets import Dataset, DatasetDict

from config import DB_URL, HF_WRITE_TOKEN


dataset_query = """
WITH unique_claims AS (
     SELECT DISTINCT
         dd.id,
         dd.claim,
         dd.evidence_sentence_id,
         dd.short_claim,
         dd.label,
         dd.evidence_wiki_url,
         dd.set_type
     FROM def_dataset dd
)
SELECT
     uq.id,
     uq.claim AS claim,
     uq.short_claim,
     uq.label,
     docs.document_id,
     docs.text,
     docs.lines,
     GROUP_CONCAT(uq.evidence_sentence_id, ';') AS evidence_lines
FROM unique_claims AS uq
JOIN documents docs ON docs.document_id = uq.evidence_wiki_url
WHERE uq.set_type = '{set_type}'
GROUP BY uq.id, docs.document_id;
"""

train_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='train'), cache_dir=None, con=DB_URL)
dev_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='dev'), cache_dir=None, con=DB_URL)
test_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='test'), cache_dir=None, con=DB_URL)

combined_datasets = DatasetDict({
      "train": train_dataset_raw,
      "dev": dev_dataset_raw,
      "test": test_dataset_raw
})

combined_datasets.push_to_hub("lukasellinger/filtered_fever_claim_verification", private=True, token=HF_WRITE_TOKEN)

# dataset_query = """
# WITH unique_claims AS (
#     SELECT DISTINCT
#         dd.id,
#         dd.claim,
#         dd.evidence_sentence_id,
#         dd.short_claim,
#         dd.label,
#         dd.evidence_wiki_url,
#         dd.set_type
#     FROM def_dataset dd
# )
# SELECT
#     uq.id,
#     uq.claim AS claim,
#     uq.short_claim,
#     uq.label,
#     docs.document_id,
#     docs.text,
#     docs.lines,
#     GROUP_CONCAT(uq.evidence_sentence_id, ';') AS evidence_lines,
#     (SELECT GROUP_CONCAT(se.evidence_lines)
#      FROM selected_evidence se
#      WHERE se.claim_id = uq.id and se.document_id = docs.document_id) AS selected_evidence_lines,
#     (SELECT GROUP_CONCAT(af.fact, '--;--')
#      FROM atomic_facts_fever_short_dissim af
#      WHERE af.claim_id = uq.id) AS atomic_facts
# FROM unique_claims AS uq
# JOIN documents docs ON docs.document_id = uq.evidence_wiki_url
# WHERE uq.set_type = '{set_type}' and uq.label != 'NOT ENOUGH INFO'
# GROUP BY uq.id, docs.document_id;
# """
#
# dataset_query1 = """
# select gdd.*, GROUP_CONCAT(af.fact, '--;--') as atomic_facts
# from squad_dataset gdd
# left join atomic_facts_squad_dissim af on af.claim_id = gdd.id and 1=1
# group by gdd.id
# """

#dataset.push_to_hub("lukasellinger/squad_claim_verification_dissim-v1", private=True, token=HF_WRITE_TOKEN)
# train_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='train'), cache_dir=None, con=DB_URL)
# dev_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='dev'), cache_dir=None, con=DB_URL)
# test_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='test'), cache_dir=None, con=DB_URL)
#
# combined_datasets = DatasetDict({
#      "train": train_dataset_raw,
#      "dev": dev_dataset_raw,
#      "test": test_dataset_raw
# })
#
# combined_datasets.push_to_hub("lukasellinger/fever_claim_verification_dissim-v1", private=True, token=HF_WRITE_TOKEN)

