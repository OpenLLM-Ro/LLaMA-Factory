import json
import os
from typing import List
import sys
import datasets
import random
random.seed(42)

non_reasoning_prompts = [
    "{0} Răspunde cu cât mai puține cuvinte posibile.",
    "{0}",
]

reasoning_prompts = [
    "{0} Răspunde la întrebare și explică-ți răspunsul pas cu pas.",
]

class CoSYN(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}],
            "images": [datasets.Value("string")]}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = "data/ro_vlm_cosyn.json"
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        data = json.load(open(filepath, "r", encoding="utf-8"))
        entry_id = -1
        for entry in data:
            # print(entry)
            added_ids = set()
            internal_iteration = 0
            while True:
                if len(set(range(len(entry["qa_pairs"]["question"]))) - added_ids) == 0:
                    break
                number_of_qas = random.randint(1, (min(3, len(entry["qa_pairs"]["question"]))))
                # print(internal_iteration, "Number of QAs:", number_of_qas)
                if number_of_qas > len(set(range(len(entry["qa_pairs"]["question"]))) - added_ids):
                    number_of_qas = len(set(range(len(entry["qa_pairs"]["question"]))) - added_ids)
                # print(internal_iteration, "Number of QAs after regularization:", number_of_qas)
                # print("Available ids:", set(range(len(entry["qa_pairs"]["question"]))) - added_ids)
                new_entry = {}
                new_entry["conversations"] = []
                for i in range(number_of_qas):
                    # print("Already added ids:", added_ids)
                    # print("Available ids:", set(range(len(entry["qa_pairs"]["question"]))) - added_ids)
                    to_add_id = random.choice(list(set(range(len(entry["qa_pairs"]["question"]))) - added_ids))
                    # print("Chosen id:", to_add_id)
                    # print()
                    added_ids.add(to_add_id)
                    if random.random() < 0.75:
                        rtype = "reasoning"
                        prompt = random.choice(reasoning_prompts)
                    else:
                        rtype = "non_reasoning"
                        prompt = random.choice(non_reasoning_prompts)
                    full_prompt = prompt.format(entry["qa_pairs"]["question"][to_add_id])
                    if i == 0:                
                        if random.random() < 0.5:
                            new_entry["conversations"].append({"from": "human", "value": "<image>\n" + full_prompt})
                        else:
                            new_entry["conversations"].append({"from": "human", "value": full_prompt + "\n<image>"})
                    else:
                        new_entry["conversations"].append({"from": "human", "value": full_prompt})
                    if rtype == "reasoning":
                        new_entry["conversations"].append({"from": "gpt", "value": entry["qa_pairs"]["explanation"][to_add_id] + " Răspuns: " + entry["qa_pairs"]["answer"][to_add_id]})
                    else:
                        new_entry["conversations"].append({"from": "gpt", "value": entry["qa_pairs"]["answer"][to_add_id]})
    
                new_entry["images"] = [os.path.join("/export/projects/nlp/datasets/rovlm/images/cosyn/", entry["images"][0])]
                internal_iteration += 1
                entry_id += 1
                yield entry_id, new_entry

            # sys.exit()


if __name__ == "__main__":

    cosyn = CoSYN()
    x = cosyn._generate_examples("cosyn/cosyn_train_full.json")
    for i in range(7):
        print(next(x))
        print()
        # print(next(x))