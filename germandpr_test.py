"""Test for german dpr dataset."""

from datasets import load_dataset

# we can concatenate train + test since we do not train on
print('hi')


CONVERSION = {"Was ist": "{} ist {}",
              "Was bezeichnet man als": "Als {} bezeichnet man {}",
              "Was bezeichnet": "{} bezeichnet {}",
              "Was bedeutet": "{} bedeuet {}",
              "Was macht": "{} macht {}",
              "Was kennzeichnet": "{} kennzeichnet{}"
              }
# we can concatenate train + test since we do not train on it
dataset = load_dataset("deepset/germandpr")['train']
selected_dataset = dataset.select_columns(["question", "answers"])
filtered_dataset = selected_dataset.filter(
    lambda example: str(example['question']).startswith(tuple(CONVERSION.keys())))

def create_fact(entry):
    for key, value in CONVERSION.items():
        if entry['question'].startswith(key):
            entity = entry['question'][len(key) + 1: -1]
            entry['fact'] = value.format(entity, entry['answers'][0])
            return entry
    return entry

filtered_dataset = filtered_dataset.map(create_fact)
print('hi')
