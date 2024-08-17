from datasets import load_dataset
from tqdm import tqdm

from config import HF_WRITE_TOKEN
from pipeline_module.evidence_fetcher import WikipediaEvidenceFetcher
from pipeline_module.pipeline import Pipeline

dataset_name = 'lukasellinger/wiki_dump_2024-08-14'
dataset = load_dataset(dataset_name).get('train')

df = dataset.to_pandas()
unique_search_words = df['search_word'].unique()

fetcher = WikipediaEvidenceFetcher()
pipeline = Pipeline(translator=None,
                    sent_connector=None,
                    claim_splitter=None,
                    evid_fetcher=fetcher,
                    evid_selector=None,
                    stm_verifier=None,
                    lang=None)

batch_size = 4
intro_ends = {}
for i in tqdm(range(0, len(unique_search_words), batch_size)):
    batch = unique_search_words[i:i + batch_size]
    batch = [{'document_search_word': entry} for entry in batch]
    intro_ends.update(pipeline.mark_summary_sents_test_batch(batch))


def add_intro_end(entry):
    entry['intro_end_sent_idx'] = intro_ends.get(entry['title'], -1)
    return entry


dataset = dataset.map(add_intro_end)
dataset.push_to_hub(dataset_name, token=HF_WRITE_TOKEN)
