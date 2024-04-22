"""Script for creating the dataset."""

import random
from typing import Tuple

from datasets import Dataset
from tqdm import tqdm

from reader import JSONLineReader
from spacy_utils import get_ent_type, check_person, recognize_definition


def check_stats(dataset: Dataset) -> dict:
    """"Iterate through the dataset and returns stats about it."""
    stats = {}
    for entry in tqdm(dataset):
        claim = str(entry.get('claim'))

        if ent_type := get_ent_type(claim):
            ent_dict = stats.setdefault(ent_type, {})
            count = ent_dict.setdefault('count', 0)
            ent_dict['count'] = count + 1
            label = ent_dict.setdefault(entry.get('label'), 0)
            ent_dict[entry.get('label')] = label + 1
            stats[ent_type] = ent_dict
        label = stats.setdefault(entry.get('label'), 0)
        stats[entry.get('label')] = label + 1

    total = len(dataset)
    for _, value in stats.items():
        if isinstance(value, dict) and value.get('count'):
            value['prop'] = round(value.get('count') / total, 2)

    return stats


def create_def_dataset(file_in: str, file_out: str, person_prop=0.1, long=True) -> Tuple[int, int]:
    """
    Create the dataset containing definition claims.
    :param file_in: .jsonl file.
    :param file_out: .jsonl file.
    :param person_prop: Proportion of person definitions to keep.
    :param long: If True, one entry contains exactly one evidence.
    :return: Tuple containing the length of definition dataset and original dataset.
    """
    reader = JSONLineReader()
    dataset_raw = reader.read(file_in)

    def_dataset = []
    for entry in tqdm(dataset_raw, desc='Processing raw'):
        claim = str(entry.get('claim'))

        claim = claim.replace("\xa0", " ")
        if claim and recognize_definition(claim):
            if check_person(claim):
                if random.random() > person_prop:  # only take 10% of persons
                    continue

            if long:
                all_evidences = entry.get('evidence')
                for evidence_list in all_evidences:
                    if len(evidence_list) > 1:  # we do not want multi-hop references
                        continue

                    evidence = evidence_list[0]
                    long_entry = {'id': entry.get('id'),
                                  'verifiable': entry.get('verifiable'),
                                  'label': entry.get('label'),
                                  'claim': claim,
                                  'evidence_annotation_id': evidence[0],
                                  'evidence_id': evidence[1],
                                  'evidence_wiki_url': evidence[2],
                                  'evidence_sentence_id': evidence[3]}
                    def_dataset.append(long_entry)
            else:
                def_dataset.append(entry)

    reader.write(file_out, def_dataset)
    return len(def_dataset), len(dataset_raw)


print(create_def_dataset(file_in='datasets/fever/train.jsonl', file_out='datasets/def_train.jsonl',
                         person_prop=0.1))
print(create_def_dataset(file_in='dataset/fever/dev.jsonl', file_out='dataset/def_dev.jsonl',
                         person_prop=0.1))
print(create_def_dataset(file_in='dataset/fever/test.jsonl', file_out='dataset/def_test.jsonl',
                         person_prop=0.1))
