"""Main evidence selection script."""
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, BigBirdModel

from config import DB_URL
from dataset.def_dataset import DefinitionDataset
from losses.supcon import SupConLoss
from models.evidence_selection_model import EvidenceSelectionModel
import torch.nn.functional as F

from utils import calc_bin_stats, plot_graph

dataset = Dataset.from_sql("""select dd.id, dd.claim, dd.label, docs.document_id, docs.text, 
                                         docs.lines, group_concat(dd.evidence_sentence_id) as evidence_lines
                                  from def_dataset dd
                                    join documents docs on docs.document_id = dd.evidence_wiki_url
                                  where set_type='train' and length(docs.text) < 100
                                  group by dd.id, evidence_annotation_id, evidence_wiki_url
                                  limit 30""",
                           con=DB_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"
#tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
#model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

model_name = 'google/bigbird-roberta-large'
model = BigBirdModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

selection_model = EvidenceSelectionModel(model).to(device)
train_dataset = DefinitionDataset(dataset, tokenizer, mode='validation', model='evidence_selection')
train_dataloader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              batch_size=10)
criterion = SupConLoss()


def convert_to_labels(similarities, labels, k=2):
    top_indices = torch.topk(similarities, k=k)[1]
    predicted = torch.zeros_like(similarities)
    predicted.scatter_(1, top_indices, 1)

    top_k_hits = labels[torch.arange(10).unsqueeze(1), top_indices]
    top_k_hits = torch.any(top_k_hits == 1, dim=1).float()

    mask = (labels != -1).flatten()
    return predicted.flatten()[mask], labels.flatten()[mask], top_k_hits


gt_labels = []
pr_labels = []
claim_lenghts = []
doc_lenghts = []
all_top_k_hits = []
for batch in tqdm(train_dataloader):
    selection_model.eval()
    with torch.no_grad():
        model_input = batch["model_input"]
        claim_embedding = selection_model(input_ids=model_input['claim_input_ids'],
                                          attention_mask=model_input['claim_attention_mask'])
        sentence_embeddings = selection_model(input_ids=model_input['input_ids'],
                                              attention_mask=model_input['attention_mask'],
                                              sentence_mask=model_input['sentence_mask'])

        loss = criterion(claim_embedding, sentence_embeddings, labels=batch['labels'])

        claim_similarities = F.cosine_similarity(claim_embedding, sentence_embeddings, dim=2)
        claim_similarities = claim_similarities.nan_to_num(nan=float('-inf'))

    predicted, true_labels, top_k_hits = convert_to_labels(claim_similarities, batch['labels'], k=3)
    gt_labels.extend(true_labels.tolist())
    pr_labels.extend(predicted.tolist())
    all_top_k_hits.extend(top_k_hits.tolist())

    if 'claim_length' in batch and 'doc_length' in batch:
        def extend_to_labels(values, labels) -> list[int]:
            mask = (labels != -1).flatten()
            values = values.unsqueeze(1).expand(-1, labels.shape[1])
            values = values.flatten()[mask]
            return values.tolist()
        claim_lenghts.extend(extend_to_labels(batch['claim_length'], batch['labels']))
        doc_lenghts.extend(extend_to_labels(batch['doc_length'], batch['labels']))

acc = accuracy_score(gt_labels, pr_labels)
recall = recall_score(gt_labels, pr_labels)
f1_weighted = f1_score(gt_labels, pr_labels, average='weighted')
f1_macro = f1_score(gt_labels, pr_labels, average='macro')
top_k_acc = sum(all_top_k_hits) / len(train_dataset)

if len(claim_lenghts) > 0:
    bin_stats = calc_bin_stats(gt_labels, pr_labels, claim_lenghts)
    print(bin_stats)
    plot_graph(list(bin_stats.keys()), [entry['acc'] for entry in bin_stats.values()],
               x_label='Claim Length', y_label='Acc')

if len(claim_lenghts) > 0:
    bin_stats = calc_bin_stats(gt_labels, pr_labels, doc_lenghts)
    print(bin_stats)
    plot_graph(list(bin_stats.keys()), [entry['acc'] for entry in bin_stats.values()],
               x_label='Doc Length', y_label='Acc')

print(acc)
print(recall)
print(f1_weighted)
print(f1_macro)
print(top_k_acc)
