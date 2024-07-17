import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    #device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

messages = [
    {"role": "system", "content": "You replace MASK tokens with the most appropriate verb, regardless of their actual relation in terms of the word knowledge."},
    {"role": "user", "content": "Fuse the following two sentence splitted by MASK. Use one of the following verbs: [be, represent, denote, refer, signify, constitute, mean, stand, imply, equal, symbolize, describe, manifest, correspond, characterize, epitomize, exemplify, embody, portray].\nInput: Kyffhäuserdenkmal MASK Symptoms that manifest directly after infection"}
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
