# Adapted from
# https://github.com/dominiksinsaarland/document-level-FEVER/blob/main/src/sentence_selection_dataset.py
"""Dataset for definitions."""

import re
from enum import Enum
from typing import Tuple, List

import torch
from torch.utils.data import Dataset

from general_utils.utils import convert_to_unicode, process_sentence_wiki


class Fact(Enum):
    """Represents the types a fact can have."""

    SUPPORTED = 0
    NOT_SUPPORTED = 1

    def to_factuality(self) -> int:
        """Convert itself to a measurement."""
        factuality = {
            Fact.SUPPORTED: 1,
            Fact.NOT_SUPPORTED: 0
            #Fact.SUPPORTS: 1,
            #Fact.REFUTES: 0,
            #Fact.NOT_ENOUGH_INFO: -1
        }
        return factuality[self]

def process_lines(lines):
    """Removes empty lines."""
    return re.sub(r'(\d+\t\n)|(\n\d+\t$)', '', lines)


def split_text(line: str) -> Tuple[str, str]:
    """Splits the text into line number and text.
    :rtype: object
    """
    tab_splits = line.split('\t')
    line_number = tab_splits[0]
    text = tab_splits[1]
    return line_number, text


def process_data(data, max_length=2000, k=3):
    """Filters text longer than max_length chars and evidence list larger than k."""
    def filter_entry(entry):
        text = process_sentence_wiki(entry['text'])
        evidence_lines = entry.get('evidence_lines')
        if evidence_lines is None:
            evidence_lines = ""
        # we do not want to have claims with evidence groups > k. There are only about 40 with more
        # than 3 lines in a group.
        return len(' '.join(text).split()) < max_length and all(
            len(group.split()) <= k for group in evidence_lines.split(';'))

    return data.filter(lambda i: filter_entry(i))


def build_attention_masks(lst: List[List], pad_token=0, attention_mask_pad=0) -> List[List]:
    """Pad lst inline with pad_token and return attention mask padded with attention_mask_pad."""
    attention_masks = []
    max_length = max(len(i) for i in lst)
    for i in lst:
        length = len(i)
        pad_length = max_length - length
        i += [pad_token] * pad_length
        attention_mask = [1] * length + [attention_mask_pad] * pad_length
        attention_masks.append(attention_mask)
    return attention_masks


def pad(lst: List[List], pad_token=0):
    """Pad lst inline with pad_token."""
    max_length = max(len(i) for i in lst)
    for i in lst:
        length = len(i)
        pad_length = max_length - length
        i += [pad_token] * pad_length


def pad_nested_lists(lst: List[List[List]], pad_token=0):
    """Pad nested lst inline with pad_token."""
    max_sentence_count = max(len(i) for i in lst)
    max_sentence_length = max(len(item) for sublist in lst for item in sublist)

    for sublist in lst:
        sublist.extend([[pad_token] for _ in range(max_sentence_count - len(sublist))])

        for subsublist in sublist:
            subsublist += [pad_token] * (max_sentence_length - len(subsublist))


class DefinitionDataset(Dataset):
    """Dataset for Definitions. One can choose for which model the dataset should be built.
    Each entry encodes the whole document at once."""

    def __init__(self, data, tokenizer=None, mode="train", model=None):
        if model in ['claim_verification', 'evidence_selection']:
            self.model = model
        else:
            pass
            #raise ValueError(
            #    f'Model needs to be "claim_verification" or "evidence_selection" but is: {model}')
        self.mode = mode
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data = process_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        if self.model == 'claim_verification':
            model_inputs, labels, hypothesis_lengths = self.get_batch_input_claim_verification(batch)
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
            claim_lengths = [len(i["claim"]) for i in batch]
            doc_lengths = [len(i["text"]) for i in batch] if self.model == 'evidence_selection' else hypothesis_lengths

            return {"model_input": model_inputs, "labels": labels, "documents": documents,
                    "evidence_lines": evidence_lines, "claim_id": claim_ids,
                    "claim_length": torch.tensor(claim_lengths), "doc_length": torch.tensor(doc_lengths)}

    def get_batch_input_claim_verification(self, batch):
        all_input_ids, all_labels, hypothesis_lengths, claim_mask = [], [], [], []

        for i, data in enumerate(batch):
            evidence_lines = set(re.split(r'[;,]',  data['evidence_lines']))
            hypothesis = ""
            lines = process_lines(data['lines'])
            for line in lines.split('\n'):
                line = process_sentence_wiki(line)
                line_number, text = split_text(line)
                if line_number not in evidence_lines:
                    continue
                hypothesis += text
                hypothesis += ' '

            if (facts := data.get('atomic_facts')) is not None:
                for fact in facts.split('--;--'):
                    claim_mask.append(i)
                    all_input_ids.append(self.tokenizer.encode(hypothesis, fact))
            else:
                all_input_ids.append(self.tokenizer.encode(hypothesis, data['claim']))
                claim_mask.append(i)

            if data['label'] == 'SUPPORTS':
                all_labels.append(Fact.SUPPORTED.to_factuality())
            else:
                all_labels.append(Fact.NOT_SUPPORTED.to_factuality())
            hypothesis_lengths.append(len(hypothesis))

        unique_claim_numbers = set(claim_mask)
        claim_masks = []
        for num in unique_claim_numbers:
            claim_masks.append([1 if val == num else 0 for val in claim_mask])

        attention_masks = build_attention_masks(all_input_ids,
                                                pad_token=self.tokenizer.pad_token_id)

        model_input = {'input_ids': torch.tensor(all_input_ids).to(self.device),
                       'attention_mask': torch.tensor(attention_masks).to(self.device),
                       'claim_mask': torch.tensor(claim_masks).to(self.device)}
        labels = torch.tensor(all_labels).to(self.device)
        return model_input, labels, hypothesis_lengths

    def get_batch_input_evidence_selection(self, batch):
        all_claim_input_ids, all_input_ids, all_labels, all_sentence_mask = [], [], [], []

        for data in batch:
            evidence_lines = set(re.split(r'[;,]',  data['evidence_lines']))

            # query = 'Represent this sentence for searching relevant passages: ' + data['claim']
            encoded_claim = self.tokenizer.encode(data['claim']) # [1:-1]  # test without cls, sep token
            lines = process_lines(data['lines'])
            labels = []
            encoded_sequence = []
            sentence_mask = []
            for line in lines.split('\n'):
                line = process_sentence_wiki(line)
                line_number, text = split_text(line)
                encoded_line = self.tokenizer.encode(text)[1:-1]
                encoded_sequence += encoded_line
                sentence_mask += [int(line_number)] * len(encoded_line)
                #sentence_mask += [int(line_number)] + [0] * (len(encoded_line) - 1)  # try only with cls token
                labels.append(1 if line_number in evidence_lines else 0)
                encoded_sequence.append(self.tokenizer.sep_token_id)
                sentence_mask.append(-1)

            unique_sentence_numbers = set(sentence_mask)
            sentence_masks = []
            for num in unique_sentence_numbers:
                if num == -1:
                    continue
                sentence_masks.append([1 if val == num else 0 for val in sentence_mask])

            all_claim_input_ids.append(encoded_claim)
            all_input_ids.append(encoded_sequence)
            all_labels.append(labels)
            all_sentence_mask.append(sentence_masks)

        claim_attention_masks = build_attention_masks(all_claim_input_ids,
                                                      pad_token=self.tokenizer.pad_token_id)
        pad(all_labels, pad_token=-1)
        pad_nested_lists(all_sentence_mask, pad_token=0)
        attention_masks = build_attention_masks(all_input_ids,
                                                pad_token=self.tokenizer.pad_token_id)

        model_input = {'claim_input_ids': torch.tensor(all_claim_input_ids).to(self.device),
                       'claim_attention_mask': torch.tensor(claim_attention_masks).to(self.device),
                       'input_ids': torch.tensor(all_input_ids).to(self.device),
                       'attention_mask': torch.tensor(attention_masks).to(self.device),
                       'sentence_mask': torch.tensor(all_sentence_mask).to(self.device)}
        labels = torch.tensor(all_labels).to(self.device)
        return model_input, labels


class SentenceContextDataset(Dataset):
    """Sentence Dataset. Each entry is a sentence with its surrounding context."""

    def __init__(self, data, tokenizer, mode="train"):
        self.mode = mode
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        data = process_data(data)
        self.data = self.build_context(data)

    @staticmethod
    def clean_lines(lines: str) -> list[tuple[str, str]]:
        """Cleans the lines and returns them in a list of tuples (line_number, text)."""
        processed_lines = []
        for line in lines.split('\n'):
            processed_line = process_sentence_wiki(line)
            line_number, text = split_text(processed_line)
            processed_lines.append((line_number, text))
        return processed_lines

    def build_context(self, data):
        new_data = []

        for entry in data:
            evidence_lines = set(re.split(r'[;,]',  data['evidence_lines']))
            lines = process_lines(entry['lines'])

            processed_lines = self.clean_lines(lines)
            for i, (line_number, line) in enumerate(processed_lines):
                new_entry = {'claim': entry['claim'],
                             'id': entry['id'],
                             'document_id': entry['document_id'],
                             'text': entry['text']}

                if line_number in evidence_lines:
                    new_entry['label'] = 1
                else:
                    new_entry['label'] = 0

                new_entry['lines'] = []
                if i > 0:
                    new_entry['lines'].append(processed_lines[i - 1][1])

                new_entry['lines'].append(processed_lines[i][1])
                new_entry['idx'] = 1 if i > 0 else 0  # index of sentence to attend to

                if i < len(processed_lines) - 1:
                    new_entry['lines'].append(processed_lines[i + 1][1])

                new_data.append(new_entry)
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        all_claim_input_ids, all_input_ids, all_labels, all_sentence_mask = [], [], [], []

        for data in batch:
            encoded_claim = self.tokenizer.encode(data['claim'])
            encoded_sequence = []
            sentence_mask = []
            for i, line in enumerate(data['lines']):
                encoded_line = self.tokenizer.encode(line)[1:-1]  # + [1]
                encoded_sequence += encoded_line

                if i == data['idx']:
                    sentence_mask += [1] * len(encoded_line)
                else:
                    sentence_mask += [0] * len(encoded_line)

                encoded_sequence.append(self.tokenizer.sep_token_id)
                sentence_mask.append(0)

            all_claim_input_ids.append(encoded_claim)
            all_input_ids.append(encoded_sequence)
            all_labels.append(data['label'])
            all_sentence_mask.append(sentence_mask)

        claim_attention_masks = build_attention_masks(all_claim_input_ids,
                                                      pad_token=self.tokenizer.pad_token_id)
        pad(all_sentence_mask, pad_token=0)
        attention_masks = build_attention_masks(all_input_ids,
                                                pad_token=self.tokenizer.pad_token_id)

        model_input = {'claim_input_ids': torch.tensor(all_claim_input_ids).to(self.device),
                       'claim_attention_mask': torch.tensor(claim_attention_masks).to(self.device),
                       'input_ids': torch.tensor(all_input_ids).to(self.device),
                       'attention_mask': torch.tensor(attention_masks).to(self.device),
                       'sentence_mask': torch.tensor(all_sentence_mask).unsqueeze(1).to(self.device)}
        labels = torch.tensor(all_labels).to(self.device)

        if self.mode == "train":
            return {"model_input": model_input, "labels": labels}
        elif self.mode == "validation":
            documents = [i["document_id"] for i in batch]
            claim_ids = [i["id"] for i in batch]
            claim_lengths = [len(i["claim"]) for i in batch]
            doc_lengths = [len(i["text"]) for i in batch]

            return {"model_input": model_input, "labels": labels, "documents": documents,
                    "claim_id": claim_ids, "claim_length": torch.tensor(claim_lengths),
                    "doc_length": torch.tensor(doc_lengths)}


class SentenceContextContrastiveDataset(SentenceContextDataset):
    """Contrastive sentence dataset. Each entry contains a single encoding for each sentence
    surrounded by its context."""

    def build_context(self, data):
        new_data = []

        for entry in data:
            new_entry = {'claim': entry['claim'],
                         'id': entry['id'],
                         'document_id': entry['document_id'],
                         'text': entry['text'],
                         'labels': [],
                         'sentences': [],
                         'indices': []}

            evidence_lines = set(re.split(r'[;,]',  data['evidence_lines']))
            lines = process_lines(entry['lines'])
            processed_lines = self.clean_lines(lines)
            for i, (line_number, line) in enumerate(processed_lines):
                if line_number in evidence_lines:
                    new_entry['labels'].append(1)
                else:
                    new_entry['labels'].append(0)

                sentence = []
                if i > 0:
                    sentence.append(processed_lines[i - 1][1])

                sentence.append(processed_lines[i][1])
                new_entry['indices'].append(1 if i > 0 else 0)  # index of sentence to attend to

                if i < len(processed_lines) - 1:
                    sentence.append(processed_lines[i + 1][1])
                new_entry['sentences'].append(sentence)
            new_data.append(new_entry)
        return new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        all_claim_input_ids, all_input_ids, all_labels, all_sentence_mask = [], [], [], []

        for data in batch:
            encoded_claim = self.tokenizer.encode(data['claim'])
            encoded_sequence, sentence_masks, labels = [], [], []
            for context, label, idx in zip(data['sentences'], data['labels'], data['indices']):
                sentence_mask = []
                for i, sentence in enumerate(context):
                    encoded_line = self.tokenizer.encode(sentence)[1:-1]  # + [1]
                    encoded_sequence += encoded_line

                    if i == idx:
                        sentence_mask += [1] * len(encoded_line)
                    else:
                        sentence_mask += [0] * len(encoded_line)

                    encoded_sequence.append(self.tokenizer.sep_token_id)
                    sentence_mask.append(0)
                sentence_masks.append(sentence_mask)
                labels.append(label)

            all_claim_input_ids.append(encoded_claim)
            all_input_ids.append(encoded_sequence)
            all_labels.append(labels)
            all_sentence_mask.append(sentence_masks)

        pad(all_labels, pad_token=-1)
        claim_attention_masks = build_attention_masks(all_claim_input_ids,
                                                      pad_token=self.tokenizer.pad_token_id)
        pad_nested_lists(all_sentence_mask, pad_token=0)
        attention_masks = build_attention_masks(all_input_ids,
                                                pad_token=self.tokenizer.pad_token_id)

        model_input = {'claim_input_ids': torch.tensor(all_claim_input_ids).to(self.device),
                       'claim_attention_mask': torch.tensor(claim_attention_masks).to(self.device),
                       'input_ids': torch.tensor(all_input_ids).to(self.device),
                       'attention_mask': torch.tensor(attention_masks).to(self.device),
                       'sentence_mask': torch.tensor(all_sentence_mask).to(self.device)}
        labels = torch.tensor(all_labels).to(self.device)

        if self.mode == "train":
            return {"model_input": model_input, "labels": labels}
        elif self.mode == "validation":
            documents = [i["document_id"] for i in batch]
            claim_ids = [i["id"] for i in batch]
            claim_lengths = [len(i["claim"]) for i in batch]
            doc_lengths = [len(i["text"]) for i in batch]

            return {"model_input": model_input, "labels": labels, "documents": documents,
                    "claim_id": claim_ids, "claim_length": torch.tensor(claim_lengths),
                    "doc_length": torch.tensor(doc_lengths)}
