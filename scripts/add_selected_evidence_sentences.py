import random

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import DB_URL
from database.db_retriever import FeverDocDB
from dataset.def_dataset import DefinitionDataset
from general_utils.utils import convert_document_id_to_word
from models.evidence_selection_model import EvidenceSelectionModel
from pipeline.pipeline import TestPipeline


def main(table):
    CREATE_SELECTED_EVIDENCE = f"""
    CREATE TABLE IF NOT EXISTS {table} (
        claim_id INTEGER,
        document_id VARCHAR,
        evidence_lines VARCHAR,
        PRIMARY KEY (claim_id, document_id)
    );
    """

    INSERT_ENTRY = f"""
    INSERT INTO {table} (claim_id, document_id, evidence_lines)
    VALUES (?, ?, ?)
    """

    with FeverDocDB() as db:
        db.write(CREATE_SELECTED_EVIDENCE)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = 'lukasellinger/evidence_selection_model-v2'
    model = AutoModel.from_pretrained('lukasellinger/evidence_selection_model-v2', trust_remote_code=True, add_pooling_layer=False, safe_serialization=True)
    selection_model = EvidenceSelectionModel(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_raw = Dataset.from_sql("""select dd.id, dd.claim, dd.short_claim, docs.text, docs.document_id, group_concat(dd.evidence_sentence_id, ';') as evidence_lines
                                          from def_dataset dd
                                            join documents docs on docs.document_id = dd.evidence_wiki_url
                                            left join selected_evidence se on (dd.id, docs.document_id) = (se.claim_id, se.document_id)
                                          where se.evidence_lines is null and docs.text != ''
                                          group by dd.id, docs.document_id
                                    """,
                                   cache_dir=None, con=DB_URL)
    dataset = DefinitionDataset(dataset_raw)

    pipeline = TestPipeline(selection_model=selection_model, selection_model_tokenizer=tokenizer)

    with FeverDocDB() as db:
        for entry in tqdm(dataset.data):
            try:
                evidences, _ = pipeline.fetch_evidence(entry['document_id'])
                claim = f"{convert_document_id_to_word(entry['document_id'])}: {entry['short_claim']}"
                selected_evidence = pipeline.select_evidence(claim, evidences)
                pr_evidence_lines = [evidence[1] for evidence in selected_evidence]
                if gold_evidence_lines := entry['evidence_lines']:
                    evidence_groups = gold_evidence_lines.split(';')
                    pr_useable = False
                    for group in evidence_groups:
                        if all(line in pr_evidence_lines for line in group.split(',')):
                            pr_useable = True
                            break
                    if not pr_useable:  # no gold evidence in predicted evidence
                        print(f"Claim predictions false: {entry['claim']}")
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
            except Exception as e:
                print(f"Exception at {entry['claim']}")


if __name__ == "__main__":
    table = 'selected_evidence'
    main(table)
