import json
import os
from typing import List
import sys
import datasets
import random
random.seed(42)

count_prompts = [
"Câte {label} sunt în imagine?",
"Câte {label} există?",
"Câte {label} se pot vedea?",
"Câte {label}?",
"câte {label} sunt în imagine?",
"câte {label} există?",
"câte {label} se pot vedea?",
"câte {label}?",
"Câți {label} sunt în imagine?",
"Câți {label} există?",
"Câți {label} se pot vedea?",
"Câți {label}?",
"câți {label} sunt în imagine?",
"câți {label} există?",
"câți {label} se pot vedea?",
"câți {label}?",
"Numără {label} în imagine.",
"Spune-mi câte {label} sunt",
"Spune-mi câte {label} sunt în imagine",
"numără {label}",
"numără fiecare {label}",
"numără {label}",
"Numără {label}.",
"Câte {label} vezi?",
"câte {label} vezi?",
"câte {label} sunt vizibile?",
"Câți {label} vezi?",
"câți {label} vezi?",
"câți {label} sunt vizibile?",
"Numără toate {label}",
"numără toate {label}",
"câte {label} sunt?",
"câți {label} sunt?",
"numără fiecare {label}",
"Numără toate {label}",
"Numără fiecare {label}",
"Care este numărul total de {label} din imagine?",
"În toată imaginea, câte {label} există?",
"Câte {label} sunt în imagine?",
"În toată imaginea, câți {label} există?",
"Câți {label} sunt în imagine?",
"Dă-mi numărul de {label} din imagine.",
"Câte {label} sunt vizibile în imagine?",
"Câte {label} sunt?",
"Câți {label} sunt vizibile în imagine?",
"Câți {label} sunt?",
"În imagine, câte {label} sunt?",
"În imagine, câți {label} sunt?",
"Object: {label}\nInstrucțiune: Câte sunt?",
"Object: {label}\nInstrucțiune: Câți sunt?",
]


class PixmoCount(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}],
            "images": [datasets.Value("string")]}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = "data/ro_vlm_pixmo_count.json"
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        data = json.load(open(filepath, "r", encoding="utf-8"))
        for entry_id, entry in enumerate(data):
            new_entry = {}
            new_entry["conversations"] = []
            prompt = random.choice(count_prompts)
            if random.random() < 0.25:
                prompt = prompt + "\nRăspundeți doar cu un număr."
            if random.random() < 0.5:
                new_entry["conversations"].append({"from": "human", "value": "<image>\n" + prompt.format(label=entry["label"])})
            else:
                new_entry["conversations"].append({"from": "human", "value": prompt.format(label=entry["label"]) + "\n<image>"})
            new_entry["conversations"].append({"from": "gpt", "value": str(entry["count"])})
            new_entry["images"] = [os.path.join("/export/projects/nlp/datasets/rovlm/", entry["images"][0])]
            yield entry_id, new_entry


if __name__ == "__main__":

    aya = PixmoCount()
    x = aya._generate_examples("data/ro_vlm_pixmo_count.json")
    for i in range(5):
        print(next(x))
        print(next(x))