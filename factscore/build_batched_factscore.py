import string


def build_prompts(self, topic, atomic_facts, knowledge_source):
    prompts = []
    for atom in atomic_facts:
        atom = atom.strip()
        if self.lm:
            retrieved = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            word = retrieved.get('word')
            passages = retrieved.get('passages')
            atom = f'{word}: {atom}'

            definition = "Answer the question about {} based on the given context.\n\n".format(
                word)
            context = ""
            for psg_idx, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"],
                                                            psg["text"].replace("<s>",
                                                                                "").replace(
                                                                "</s>", ""))
            definition += context.strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(),
                                                                      atom.strip())
            prompts.append(prompt)


def build_prompt():
    pass
