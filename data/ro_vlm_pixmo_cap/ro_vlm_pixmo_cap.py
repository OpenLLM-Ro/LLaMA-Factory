import json
import os
from typing import List
import sys
import datasets
import random
random.seed(42)

caption_prompts = ['Descrie această imagine.',
'Descrie această imagine',
'descrie imaginea',
'Scrie o descriere lungă a acestei imagini.',
'adaugă o descriere imaginii',
'Descrie',
'descrie',
'Construiește o descriere lungă pentru această imagine',
'Generează o descriere',
'Creează o descriere detaliată',
'Scrie o descriere lungă',
'Descrie această imagine în detaliu',
'Descrie asta',
'descrie asta',
'Adaugă o descriere',
'Ce se poate vedea în această imagine?',
'Ce vezi în imagine?',
'Uită-te cu atenție la această fotografie și apoi spune-mi despre ea în detaliu',
'Scrie o descriere lungă a acestei imagini',
'Spune-mi despre această imagine.',
'Scrie un paragraf despre această imagine.',
'Uită-te cu atenție la această imagine și apoi descrie-o în detaliu',
'Generează o descriere lungă despre această imagine.']


class PixmoCap(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}],
            "images": [datasets.Value("string")]}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = "data/ro_vlm_pixmo_cap.json"
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
            new_entry["conversations"].append({"from": "gpt", "value": entry["caption"]})
            new_entry["images"] = ["/export/projects/nlp/datasets/rovlm/images/pixmo_cap_ro/{0:06d}.jpg".format(entry["id"])]
            assert "conversations" in new_entry
            yield entry_id, new_entry


if __name__ == "__main__":

    aya = PixmoCap()
    x = aya._generate_examples("data/ro_vlm_pixmo_cap.json")
    print(next(x))
    print(next(x))