from datasets import load_dataset
from tqdm import tqdm

from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.evidence_selector import ModelEvidenceSelector
from pipeline_module.pipeline import Pipeline
from pipeline_module.sentence_connector import PhiSentenceConnector
from pipeline_module.translator import OpusMTTranslator

import torch
import numpy as np
import random
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

finetuned_selection_model = 'lukasellinger/evidence_selection_model-v3'
base_model = 'Snowflake/snowflake-arctic-embed-m-long'
model = ModelEvidenceSelector(model_name=finetuned_selection_model)

translator = OpusMTTranslator()

# Sentence Connectors
phi_sentence_connector = PhiSentenceConnector()

# Evidence Fetcher
offline_evid_fetcher = WikipediaEvidenceFetcher()

pipeline = Pipeline(translator=translator,
                            sent_connector=phi_sentence_connector,
                            claim_splitter=None,
                            evid_fetcher=offline_evid_fetcher,
                            evid_selector=model,
                            stm_verifier=None,
                            lang='de')

dataset = load_dataset('lukasellinger/german_dpr-claim_verification', split='test')
dataset = dataset.to_list()

outputs, gt_labels, pr_labels = [], [], []
not_in_wiki = 0
batch_size = 4
batch = dataset[12:14]
output1 = pipeline.verify_test_select(batch, only_intro=True,
                                         max_evidence_count=3, top_k=3)
output2 = pipeline.verify_test_select(batch, only_intro=True,
                                         max_evidence_count=3, top_k=3)

output3 = pipeline.verify_test_select(batch, only_intro=True,
                                         max_evidence_count=3, top_k=3)

if output1 != output2:
    print('hi')
    output1 = pipeline.verify_test_select(batch, only_intro=True,
                                              max_evidence_count=3, top_k=3)
    output2 = pipeline.verify_test_select(batch, only_intro=True,
                                              max_evidence_count=3, top_k=3)