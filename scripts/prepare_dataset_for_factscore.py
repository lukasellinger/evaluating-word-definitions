from datasets import load_dataset

from general_utils.reader import JSONLineReader


def main(dataset_name, output_file):
    dataset = load_dataset(dataset_name)['train']

    datalist = []
    for entry in dataset:
        datalist.append({
            'id': entry['id'],
            'topic': (entry['word'], entry['english_word'] if entry.get('english_word') else entry['word']),
            'output': entry['english_claim'] if entry.get('english_claim') else entry['claim'],
            'label': entry['label'],
            'annotations': [{'model-atomic-facts': [{'text': fact} for fact in entry['atomic_facts'].split('--;--')]}]
        })
    JSONLineReader().write(output_file, datalist)


if __name__ == "__main__":
    dataset_name = 'lukasellinger/german_claim_verification_dissim-v1'
    output_file = 'german_claim_verification_dissim-v1-factscore.jsonl'
    main(dataset_name, output_file)
