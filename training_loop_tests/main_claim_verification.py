"""Main claim verification script."""
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import DB_URL
from dataset.def_dataset import DefinitionDataset
from models.claim_verification_model import ClaimVerificationModel
from general_utils.utils import calc_bin_stats, plot_graph
from training_loop_tests.utils import plot_stats

dataset = Dataset.from_sql("""select dd.id, dd.claim as claim, dd.label, docs.document_id, docs.text, 
                                         docs.lines, group_concat(dd.evidence_sentence_id) as evidence_lines
                                  from def_dataset dd
                                    join documents docs on docs.document_id = dd.evidence_wiki_url
                                    join claim_translations ct on dd.id = ct.claim_id
                                  where set_type='train' and length(docs.text) < 800
                                  group by dd.id, evidence_annotation_id, evidence_wiki_url
                                  limit 1000""",
                           con=DB_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

verification_model = ClaimVerificationModel(model).to(device)
train_dataset = DefinitionDataset(dataset, tokenizer, mode='validation', model='claim_verification')
train_dataloader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              batch_size=10)

gt_labels = []
pr_labels = []
claim_lenghts = []
hypothesis_lengths = []
for batch in tqdm(train_dataloader):
    verification_model.eval()
    with torch.no_grad():
        model_input = batch["model_input"]
        output = verification_model(input_ids=model_input['input_ids'],
                                    attention_mask=model_input['attention_mask'])
    predicted = torch.softmax(output["logits"], dim=-1)
    predicted = torch.argmax(predicted, dim=-1)

    gt_labels.extend(batch['labels'].tolist())
    pr_labels.extend(predicted.tolist())

    if 'claim_length' in batch and 'doc_length' in batch:
        claim_lenghts.extend(batch['claim_length'])
        hypothesis_lengths.extend(batch['doc_length'])

acc = accuracy_score(gt_labels, pr_labels)
f1_weighted = f1_score(gt_labels, pr_labels, average='weighted')
f1_macro = f1_score(gt_labels, pr_labels, average='macro')

plot_stats(claim_lenghts, hypothesis_lengths, 'Hypothesis Length', gt_labels, pr_labels)

print(acc)
print(f1_weighted)
print(f1_macro)
