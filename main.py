"""Main script."""
import torch
from datasets import Dataset
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
                                  where set_type='train' and length(docs.lines) < 500
                                  group by dd.id, evidence_annotation_id, evidence_wiki_url""",
                           con=DB_URL)

#tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
#model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

model = BigBirdModel.from_pretrained("google/bigbird-roberta-large")
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-roberta-large")

selection_model = EvidenceSelectionModel(model)
train_dataset = DefinitionDataset(dataset, tokenizer, mode='train', model='evidence_selection')
train_dataloader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              batch_size=10)


def evaluate_accuracy(labels, similarities, unique_sents):
    #top_indices = torch.topk(similarities, k=2)[1]
    #t = (batch['labels'].view(-1).unsqueeze(-1) == unique_sents.unsqueeze(0)).reshape(10, 2, 5)
    ##indices = torch.argmax(t.float(), dim=-1)
    #indices[torch.sum(t, dim=-1) == 0] = -1
    #return multilabel_accuracy(input=top_indices, target=indices, criteria='overlap').item()
    top_k_similarities_idxs = torch.topk(similarities, k=2)[1]

    # TODO: get rid of for loop
    accuracies = []
    for label, top_k_similarity_idx in zip(labels, top_k_similarities_idxs):
        masks_label_idx = (label.unsqueeze(1) == unique_sents.unsqueeze(0)).nonzero()[:, 1]
        accuracies.append(torch.all(torch.isin(masks_label_idx, top_k_similarity_idx)).float().item())

    return sum(accuracies) / len(accuracies)


accuracies = []
for batch in tqdm(train_dataloader):
    selection_model.train()
    model_input = batch["model_input"]
    claim_embedding, _ = selection_model(input_ids=model_input['claim_input_ids'],
                                         attention_mask=model_input['claim_attention_mask'])
    sentence_embeddings, sentence_mask_ordering = selection_model(input_ids=model_input['input_ids'], attention_mask=model_input['attention_mask'], sentence_mask=model_input['sentence_mask'])
    similarities = F.cosine_similarity(claim_embedding, sentence_embeddings, dim=2)
    similarities = similarities.nan_to_num(nan=float('-inf'))

    accuracies.append(evaluate_accuracy(batch['labels'], similarities, sentence_mask_ordering))

print(sum(accuracies) / len(accuracies))
