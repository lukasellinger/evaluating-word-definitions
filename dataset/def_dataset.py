# Adapted from
# https://github.com/dominiksinsaarland/document-level-FEVER/blob/main/src/sentence_selection_dataset.py
"""Dataset for definitions."""

import re
from enum import Enum

import torch
from torch.utils.data import Dataset

from utils import convert_to_unicode


class Fact(Enum):
    SUPPORTED = 1
    REFUTED = 2
    NOT_ENOUGH_INFO = 3  # TODO data needed, FEVER has empty documents


def process_sentence(sentence):
    """Converts characters to their original representation in a sentence."""
    sentence = convert_to_unicode(sentence)
    sentence = re.sub(" -LSB-.*?-RSB-", " ", sentence)
    sentence = re.sub(" -LRB- -RRB- ", " ", sentence)
    sentence = re.sub("-LRB-", "(", sentence)
    sentence = re.sub("-RRB-", ")", sentence)
    sentence = re.sub("-COLON-", ":", sentence)
    sentence = re.sub("_", " ", sentence)
    sentence = re.sub(r"\( *\,? *\)", "", sentence)
    sentence = re.sub(r"\( *[;,]", "(", sentence)
    sentence = re.sub("--", "-", sentence)
    sentence = re.sub("``", '"', sentence)
    sentence = re.sub("''", '"', sentence)
    return sentence


def process_lines(lines):
    """Removes empty lines."""
    return re.sub(r'(\d\t\n)|(\n\d\t$)', '', lines)


class DefinitionDataset(Dataset):
    def __init__(self, data, tokenizer, mode="train", model=None):
        if model in ['claim_verification', 'evidence_selection']:
            self.model = model
        else:
            raise ValueError(
                f'Model needs to be "claim_verification" or "evidence_selection" but is: {model}')
        self.mode = mode
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        if self.model == 'claim_verification':
            model_inputs, labels = self.get_batch_input_claim_verification(batch)
        elif self.model == 'evidence_selection':
            model_inputs, labels = self.get_batch_input_evidence_selection(batch)
        else:
            raise ValueError(
                f'Model needs to be "claim_verification" or "evidence_selection" but is: {self.model}')

        if self.mode == "train":
            return {"model_input": model_inputs, "labels": labels}
        elif self.mode == "validation":
            documents = [i["document_id"] for i in batch]
            evidence_lines = [i["evidence_lines"] for i in batch]
            claim_ids = [i["id"] for i in batch]
            return {"model_input": model_inputs, "labels": labels, "documents": documents,
                    "evidence_lines": evidence_lines, "claim_id": claim_ids}

    def get_batch_input_claim_verification(self, batch):
        all_input_ids, all_labels = [], []

        for data in batch:
            evidence_lines = data['evidence_lines'].split(',')
            # TODO check what to do with claim in NLI model, which NLI model, ...
            encoded_claim = self.tokenizer.encode(data['claim'])
            lines = process_lines(data['lines'])
            encoded_sequence = []
            for line in lines.split('\n'):
                line = process_sentence(line)
                line_number = line.split('\t')[0]
                if line_number not in evidence_lines:
                    continue

                line = line.lstrip(f'{line_number}\t')
                encoded_line = self.tokenizer.encode(line)[1:-1]  # + [1]
                encoded_sequence += encoded_line
                encoded_sequence.append(self.tokenizer.sep_token_id)

            all_input_ids.append(encoded_sequence)
            all_labels.append(Fact[data['label']].value)
        attention_masks = self.build_attention_masks(all_input_ids,
                                                     pad_token=self.tokenizer.pad_token_id)

        model_input = {'input_ids': torch.tensor(all_input_ids),
                       'attention_mask': torch.tensor(attention_masks)}
        labels = torch.tensor(all_labels)
        return model_input, labels

    def get_batch_input_evidence_selection(self, batch):
        all_claim_input_ids, all_input_ids, all_labels, all_sentence_mask = [], [], [], []

        for data in batch:
            evidence_lines = data['evidence_lines'].split(',')
            encoded_claim = self.tokenizer.encode(data['claim'])
            lines = process_lines(data['lines'])
            labels = []
            encoded_sequence = []
            sentence_mask = []
            for line in lines.split('\n'):
                line = process_sentence(line)
                line_number = line.split('\t')[0]
                line = line.lstrip(f'{line_number}\t')
                encoded_line = self.tokenizer.encode(line)[1:-1]  # + [1]
                encoded_sequence += encoded_line
                sentence_mask += [int(line_number)] * len(encoded_line)

                if line_number in evidence_lines:
                    labels.append(int(line_number))
                encoded_sequence.append(self.tokenizer.sep_token_id)
                sentence_mask.append(-1)

            all_claim_input_ids.append(encoded_claim)
            all_input_ids.append(encoded_sequence)
            all_labels.append(labels)
            all_sentence_mask.append(sentence_mask)

        claim_attention_masks = self.build_attention_masks(all_claim_input_ids,
                                                           pad_token=self.tokenizer.pad_token_id)
        self.pad(all_labels, pad_token=-2)
        attention_masks = []
        max_length = max(len(i) for i in all_input_ids)
        for input_ids, sentence_mask in zip(all_input_ids, all_sentence_mask):
            length = len(input_ids)
            pad_length = max_length - length
            input_ids += [self.tokenizer.pad_token_id] * pad_length
            sentence_mask += [-1] * pad_length
            attention_mask = [1] * length + [0] * pad_length
            attention_masks.append(attention_mask)

        model_input = {'claim_input_ids': torch.tensor(all_claim_input_ids),
                       'claim_attention_mask': torch.tensor(claim_attention_masks),
                       'input_ids': torch.tensor(all_input_ids),
                       'attention_mask': torch.tensor(attention_masks),
                       'sentence_mask': torch.tensor(all_sentence_mask)}
        labels = torch.tensor(all_labels)
        return model_input, labels

    @staticmethod
    def build_attention_masks(list_of_lists, pad_token=0, attention_mask_pad=0):
        attention_masks = []
        max_length = max(len(i) for i in list_of_lists)
        for i in list_of_lists:
            length = len(i)
            pad_length = max_length - length
            i += [pad_token] * pad_length
            attention_mask = [1] * length + [attention_mask_pad] * pad_length
            attention_masks.append(attention_mask)
        return attention_masks

    @staticmethod
    def pad(list_of_lists, pad_token=0):
        max_length = max(len(i) for i in list_of_lists)
        for i in list_of_lists:
            length = len(i)
            pad_length = max_length - length
            i += [pad_token] * pad_length
