import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import HF_WRITE_TOKEN

torch.random.manual_seed(0)


def get_prompt(sent1: str, sent2: str):
    return [
        {"role": "system",
         "content": "You replace MASK tokens with the most appropriate verb, regardless of their actual relation in terms of the word knowledge."},
        {"role": "user",
         "content": f"Fuse the following two sentence splitted by MASK. Use one of the following verbs: [be, represent, denote, refer, signify, constitute, mean, stand, imply, equal, symbolize, describe, manifest, correspond, characterize, epitomize, exemplify, embody, portray].\nInput: {sent1} MASK {sent2}"}
    ]


def main(dataset_name, pipe):
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    dataset = load_dataset(dataset_name).get('train')
    connected_claims = []
    for entry in tqdm(dataset):
        word = entry.get('document_search_word')
        claim = entry.get('english_claim', entry['claim'])
        output = pipe(get_prompt(word, claim), **generation_args)
        model_answer = output[0]['generated_text']
        connected_claims.append(model_answer)

    dataset = dataset.add_column('connected_claim', connected_claims)
    dataset.push_to_hub(dataset_name, private=True, token=HF_WRITE_TOKEN)

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        # device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    main('lukasellinger/german_claim_verification_dissim-v1', pipe)
