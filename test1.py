import torch
from datasets import Dataset
from peft import AutoPeftModelForFeatureExtraction
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AutoTokenizer

from config import DB_URL
from dataset.def_dataset import Fact
from models.evidence_selection_model import EvidenceSelectionModel
from pipeline.pipeline import TestPipeline

dataset_query = """
select dd.id, docs.document_id, dd.claim, dd.label
from def_dataset dd
    join documents docs on docs.document_id = dd.evidence_wiki_url
    join atomic_facts af on af.claim_id = dd.id
where set_type='{set_type}' -- and length(claim) < 50 and length(docs.text) < 400
group by dd.id, evidence_annotation_id, evidence_wiki_url
limit 20
"""

dataset = Dataset.from_sql(dataset_query.format(set_type='dev'), con=DB_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'google/bigbird-roberta-large'
model = AutoPeftModelForFeatureExtraction.from_pretrained('selection_model_intermediate_04-30_09-40')

selection_model = EvidenceSelectionModel(model).to(device)
selection_model_tokenizer = AutoTokenizer.from_pretrained(model_name)

# still using base
verification_model=None
verification_model_tokenizer=None

test_pipeline = TestPipeline(selection_model=selection_model, selection_model_tokenizer=selection_model_tokenizer)

pr_labels = []
gt_labels = []
for entry in tqdm(dataset):
    factuality = test_pipeline.verify(entry['document_id'], entry['claim'])
    pr_labels.extend([fact.to_factuality() for fact in factuality])
    gt_labels += [Fact[entry['label']].to_factuality()] * len(factuality)

print(classification_report(gt_labels, pr_labels, zero_division=0))