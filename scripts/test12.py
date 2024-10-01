import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import HF_WRITE_TOKEN

model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

checkpoint_path = 'path_to_your_checkpoint_file.pth'  # Specify the path to your checkpoint file
checkpoint = torch.load(checkpoint_path)

# Load the model state
model.load_state_dict(checkpoint['model'])
model.push_to_hub('lukasellinger/claim_verification_model-keep', private=True)