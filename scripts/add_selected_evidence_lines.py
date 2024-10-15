import random

from datasets import load_dataset
from tqdm import tqdm

from dataset.def_dataset import split_text, process_lines
from general_utils.utils import process_sentence_wiki
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.pipeline import FeverPipeline

model_name = 'lukasellinger/evidence_selection_model-v4'
dataset_dict = load_dataset('lukasellinger/fever_claim_verification_dissim-v1')

evid_selector = ModelEvidenceSelector(model_name, evidence_selection='top', min_similarity=0)

for name, dataset in dataset_dict.items():
    selected_evidence_lines = []
    for entry in tqdm(dataset):
        try:
            evidence = FeverPipeline.prepare_evid(entry)
            evids_batch = evid_selector.select_evidences({'text': entry['claim']}, evidence)

            pr_evidence_lines = [evidence.get('line_idx') for evidence in evids_batch]
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
            selected_evidence_lines.append(','.join(pr_evidence_lines))
        except Exception as e:
            print(f"Exception at {entry['claim']}")
    dataset = dataset.remove_columns('selected_evidence_lines')
    dataset = dataset.add_column('selected_evidence_lines', selected_evidence_lines)
    dataset_dict[name] = dataset

dataset_dict.push_to_hub('lukasellinger/fever_claim_verification_dissim-v1', private=True)