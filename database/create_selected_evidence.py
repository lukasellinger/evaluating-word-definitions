import random

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import DB_URL
from database.db_retriever import FeverDocDB
from dataset.def_dataset import DefinitionDataset
from models.evidence_selection_model import EvidenceSelectionModel
from pipeline.pipeline import TestPipeline

CREATE_SELECTED_EVIDENCE = """
CREATE TABLE IF NOT EXISTS selected_evidence (
    claim_id INTEGER,
    document_id INTEGER,
    evidence_lines VARCHAR,
    PRIMARY KEY (claim_id, document_id)
);
"""

INSERT_ENTRY = """
INSERT INTO selected_evidence (claim_id, document_id, evidence_lines)
VALUES (?, ?, ?)
"""

with FeverDocDB() as db:
    db.write(CREATE_SELECTED_EVIDENCE)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'Snowflake/snowflake-arctic-embed-m-long'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, add_pooling_layer=False, safe_serialization=True)
selection_model = EvidenceSelectionModel(model).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset_raw = Dataset.from_sql("""select distinct dd.id, dd.claim, docs.document_id, group_concat(dd.evidence_sentence_id, ';') as gold_evidence_lines
                                      from def_dataset dd
                                        join documents docs on docs.document_id = dd.evidence_wiki_url
                                        group by dd.id, evidence_annotation_id, evidence_wiki_url
                                """,
                               con=DB_URL)
dataset = DefinitionDataset(dataset_raw)

selection_model.eval()
pipeline = TestPipeline(selection_model=selection_model, selection_model_tokenizer=tokenizer)

with FeverDocDB() as db:
    for entry in tqdm(dataset.data):
        evidences = pipeline.fetch_evidence(entry['document_id'])
        selected_evidence = pipeline.select_evidence(entry['claim'], evidences)
        pr_evidence_lines = [evidence[1] for evidence in selected_evidence]
        if gold_evidence_lines := entry['gold_evidence_lines']:
            evidence_groups = gold_evidence_lines.split(';')
            pr_useable = False
            for group in evidence_groups:
                if all(line in pr_evidence_lines for line in group.split(',')):
                    pr_useable = True
                    break
            if not pr_useable:  # no gold evidence in predicted evidence
                builded_ev_lines = []
                while len(builded_ev_lines) < 3:
                    if len(evidence_groups) == 0:
                        while len(builded_ev_lines) < 3:
                            line = pr_evidence_lines.pop()
                            if line not in builded_ev_lines:
                                builded_ev_lines.append(line)
                    else:
                        group = random.choice(evidence_groups)
                        evidence_groups.remove(group)
                        group = group.split(',')
                        if len(builded_ev_lines) + len(group) <= 3:
                            builded_ev_lines.extend(group)
                pr_evidence_lines = builded_ev_lines

        db.write(INSERT_ENTRY, (entry['id'], entry['document_id'], ','.join(pr_evidence_lines)))
