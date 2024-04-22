"""Main script."""
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, BigBirdModel

from config import DB_URL
from dataset.def_dataset import DefinitionDataset
from models.evidence_selection_model import EvidenceSelectionModel
import torch.nn.functional as F


dataset = Dataset.from_sql("""select dd.id, dd.claim, dd.label, docs.document_id,docs.lines, 
                                         group_concat(dd.evidence_sentence_id) as evidence_lines
                                  from def_dataset dd
                                    join documents docs on docs.document_id = dd.evidence_wiki_url
                                  where set_type='train' and length(docs.lines) < 800
                                  group by dd.id, evidence_annotation_id, evidence_wiki_url""",
                           con=DB_URL)

device = "cuda" if torch.cuda.is_available() else "cpu"
#tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
#model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

model = BigBirdModel.from_pretrained("google/bigbird-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-large")

selection_model = EvidenceSelectionModel(model).to(device)
train_dataset = DefinitionDataset(dataset, tokenizer, mode='train', model='evidence_selection')
train_dataloader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              batch_size=10)


def convert_to_labels(similarities, labels, k=2):
    top_indices = torch.topk(similarities, k=k)[1]
    predicted = torch.zeros_like(similarities)
    predicted.scatter_(1, top_indices, 1)

    mask = (labels != -1).flatten()
    return predicted.flatten()[mask], labels.flatten()[mask]


gt_labels = []
pr_labels = []
for batch in tqdm(train_dataloader):
    selection_model.train()
    model_input = batch["model_input"]
    claim_embedding = selection_model(input_ids=model_input['claim_input_ids'],
                                      attention_mask=model_input['claim_attention_mask'])
    sentence_embeddings = selection_model(input_ids=model_input['input_ids'],
                                          attention_mask=model_input['attention_mask'],
                                          sentence_mask=model_input['sentence_mask'])
    claim_similarities = F.cosine_similarity(claim_embedding, sentence_embeddings, dim=2)
    claim_similarities = claim_similarities.nan_to_num(nan=float('-inf'))

    predicted, true_labels = convert_to_labels(claim_similarities, batch['labels'], k=2)
    gt_labels.extend(true_labels.tolist())
    pr_labels.extend(predicted.tolist())


acc = accuracy_score(gt_labels, pr_labels)
f1_weighted = f1_score(gt_labels, pr_labels, average='weighted')
f1_macro = f1_score(gt_labels, pr_labels, average='macro')
