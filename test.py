from datetime import datetime

import torch
import gc
import numpy as np
import transformers

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from transformers import AutoTokenizer, BigBirdModel, get_linear_schedule_with_warmup
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast

from config import DB_URL
from dataset.def_dataset import DefinitionDataset
from models.evidence_selection_model import EvidenceSelectionModel
from losses.supcon import SupConLoss
import torch.nn.functional as F


def convert_to_labels(similarities, labels, k=2):
    top_indices = torch.topk(similarities, k=min(k, similarities.size(1)))[1]
    predicted = torch.zeros_like(similarities)
    predicted.scatter_(1, top_indices, 1)

    top_k_hits = labels[torch.arange(labels.size(0)).unsqueeze(1), top_indices]
    top_k_hits = torch.any(top_k_hits == 1, dim=1).float()

    mask = (labels != -1).flatten()
    return predicted.flatten()[mask], labels.flatten()[mask], top_k_hits


def evaluate(ev_model, dataloader, loss_function):
    gt_labels = []
    pr_labels = []
    all_top_k_hits = []
    all_loss = []
    for batch in tqdm(dataloader):
        ev_model.eval()
        with torch.no_grad():
            model_input = batch["model_input"]
            claim_embedding = ev_model(input_ids=model_input['claim_input_ids'],
                                       attention_mask=model_input['claim_attention_mask'])
            sentence_embeddings = ev_model(input_ids=model_input['input_ids'],
                                           attention_mask=model_input['attention_mask'],
                                           sentence_mask=model_input['sentence_mask'])

            loss = loss_function(claim_embedding, sentence_embeddings, labels=batch['labels'])
            claim_similarities = F.cosine_similarity(claim_embedding, sentence_embeddings, dim=2)
            claim_similarities = claim_similarities.nan_to_num(nan=float('-inf'))
        predicted, true_labels, top_k_hits = convert_to_labels(claim_similarities, batch['labels'],
                                                               k=3)
        gt_labels.extend(true_labels.tolist())
        pr_labels.extend(predicted.tolist())
        all_top_k_hits.extend(top_k_hits.tolist())
        all_loss.append(loss)
    loss = sum(all_loss) / len(all_loss)
    top_k_acc = sum(all_top_k_hits) / len(all_top_k_hits)

    return loss.item(), top_k_acc, classification_report(gt_labels, pr_labels)


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'google/bigbird-roberta-large'
model = BigBirdModel.from_pretrained(model_name)

# Add all lora compatible modules
target_modules = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, transformers.pytorch_utils.Conv1D)):
        target_modules.append(name)

peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=64, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

selection_model = EvidenceSelectionModel(model).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset_query = """
select dd.id, dd.claim, dd.label, docs.document_id, docs.text, 
       docs.lines, group_concat(dd.evidence_sentence_id) as evidence_lines
from def_dataset dd
    join documents docs on docs.document_id = dd.evidence_wiki_url
where set_type='{set_type}' and length(claim) < 50 and length(docs.text) < 400
group by dd.id, evidence_annotation_id, evidence_wiki_url
limit 10
"""

train_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='train'), con=DB_URL)
dev_dataset_raw = Dataset.from_sql(dataset_query.format(set_type='dev'), con=DB_URL)

train_dataset = DefinitionDataset(train_dataset_raw, tokenizer, mode='train', model='evidence_selection')
train_dataloader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=train_dataset.collate_fn,
                              batch_size=2)
dev_dataset = DefinitionDataset(dev_dataset_raw, tokenizer, mode='train', model='evidence_selection')
dev_dataloader = DataLoader(train_dataset, shuffle=True,
                              collate_fn=dev_dataset.collate_fn,
                              batch_size=2)

#warmup_steps = 0
#t_total = int(len(train_dataloader) * args.num_epochs / args.gradient_accumulation_steps)

optimizer = optim.AdamW(selection_model.parameters(), lr=1e-20)
#scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
criterion = SupConLoss(temperature=0.17)

timestamp = datetime.now().strftime("%m-%d_%H-%M")

num_epochs = 5
patience = 4
gradient_accumulation = 16  # 2048
trace_train = []
trace_val = []

selection_model.zero_grad()
use_amp = True
scaler = GradScaler(enabled=use_amp, init_scale=1)

best_loss = np.inf
for epoch in range(num_epochs):
    bar_desc = "Epoch %d of %d | Iteration" % (epoch + 1, num_epochs)
    train_iterator = tqdm(train_dataloader, desc=bar_desc)

    train_loss = 0
    print('Train ...')
    for step, batch in enumerate(train_iterator):
        selection_model.train()
        model_input = batch["model_input"]

        with autocast():
            claim_embedding = selection_model(input_ids=model_input['claim_input_ids'],
                                              attention_mask=model_input['claim_attention_mask'])
            sentence_embeddings = selection_model(input_ids=model_input['input_ids'],
                                                  attention_mask=model_input['attention_mask'],
                                                  sentence_mask=model_input['sentence_mask'])

            loss = criterion(claim_embedding, sentence_embeddings, labels=batch['labels'])
            train_loss += loss.detach().item()
            loss = (loss / gradient_accumulation)

        scaler.scale(loss).backward()

        total_norm = 0
        for name, param in selection_model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1 / 2)

        # Print gradient norm
        print(f'Gradient norm: {total_norm}')

        if (step + 1) % gradient_accumulation == 0:
            scaler.unscale_(optimizer)
            count = 0
            scaler.step(optimizer)
            scaler.update()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()

    trace_train.append(train_loss / len(train_dataloader))
    # validation
    with torch.no_grad():
        val_loss, val_top_k_acc, report = evaluate(selection_model, dev_dataloader, criterion)
        trace_val.append(val_loss)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {(train_loss / len(train_dataloader)):.4f}, Validation Loss: {val_loss:.4f}')
        print(f'Validation top k acc: {val_top_k_acc:.4f}')
        print(report)

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.cpu() for key, value in
                          selection_model.state_dict().items()}
            selection_model.save(f'selection_model_intermediate_{timestamp}')
        else:
            if epoch >= best_epoch + patience:
                break

selection_model.load_state_dict(best_state)
selection_model.save(f'selection_model_{timestamp}')

plt.plot(trace_train, label='train')
plt.plot(trace_val, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)