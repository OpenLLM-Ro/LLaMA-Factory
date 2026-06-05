import json
import os
from typing import List
import sys
import datasets
import random


caption_prompts = ['Descrie pe scurt această imagine.',
'descrie pe scurt imaginea',
'Descrie această imagine.',
'Scrie o descriere scurtă a acestei imagini.',
'adaugă o descriere scurtă imaginii',
'Descrie succint',
'Construiește o descriere scurtă pentru această imagine',
'Generează o descriere',
]

_URL = "https://huggingface.co/datasets/andreidima/Flickr30K-RoQA/blob/main/data/"
_URLS = {
    "train": [
        _URL + "train-00001-of-00008.parquet",
        _URL + "train-00002-of-00008.parquet",
        _URL + "train-00003-of-00008.parquet",
        _URL + "train-00004-of-00008.parquet",
        _URL + "train-00005-of-00008.parquet",
        _URL + "train-00006-of-00008.parquet",
        _URL + "train-00007-of-00008.parquet"
    ],
    # "test": [
    #     _URL + "test-00000-of-00001.parquet",
    # ],
    # "validation": [
        # _URL + "validation-00000-of-00001.parquet",
    # ],
}

class Flickr30KQA(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}],
            "images": [datasets.Value("string")]}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        # file_paths = dl_manager.download(_URLS["train"])
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": ""})]

    def _generate_examples(self, filepath: str):
        ds = datasets.load_dataset("andreidima/Flickr30K-RoQA", split="train")
        for entry_id, entry in enumerate(ds):
            # if not os.path.exists("/home/mihai/fep/llama-factory/LLaMA-Factory/data/images/flickr30k_ro/{0}".format(entry["file_name"])):
                # continue
            new_entry = {}
            new_entry["conversations"] = []
            prompt = random.choice(caption_prompts)
            if random.random() < 0.5:
                new_entry["conversations"].append({"from": "human", "value": "<image>\n" + prompt})
            else:
                new_entry["conversations"].append({"from": "human", "value": prompt + "\n<image>"})
            caption = random.choice(entry["caption"])
            new_entry["conversations"].append({"from": "gpt", "value": caption})
            new_entry["images"] = ["/export/projects/nlp/datasets/rovlm/images/flickr30k_ro/{0}".format(entry["file_name"])]
            assert "conversations" in new_entry
            yield entry_id, new_entry
