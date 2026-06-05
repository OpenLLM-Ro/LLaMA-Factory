import json
import os
from typing import List
import sys
import datasets
import random
random.seed(42)

caption_prompts = ['Extrage întocmai textul și doar textul din această imagine, rezolvă task-ul de OCR.',
'Extrage textul și doar textul din această imagine.',
'Extrage textul din această imagine.',
'Extrage textul'
'Ce scrie în această imagine?',
'Ce text este în această imagine?',
'Ce text se află în această imagine?']


class Finepdfs(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}],
            "images": [datasets.Value("string")]}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = "data/ro_vlm_finepdfs.json"
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        data = json.load(open(filepath, "r", encoding="utf-8"))
        for entry_id, entry in enumerate(data):
            new_entry = {}
            new_entry["conversations"] = []
            prompt = random.choice(caption_prompts)
            if random.random() < 0.5:
                new_entry["conversations"].append({"from": "human", "value": "<image>\n" + prompt})
            else:
                new_entry["conversations"].append({"from": "human", "value": prompt + "\n<image>"})
            new_entry["conversations"].append({"from": "gpt", "value": entry["text"]})
            new_entry["images"] = [os.path.join("/export/projects/nlp/datasets/rovlm/", entry["images"][0])]
            assert "conversations" in new_entry
            yield entry_id, new_entry


if __name__ == "__main__":

    fp = Finepdfs()
    x = fp._generate_examples("data/ro_vlm_finepdfs.json")
    print(next(x))
    print(next(x))