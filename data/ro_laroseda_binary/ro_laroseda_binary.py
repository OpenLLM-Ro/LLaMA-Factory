import json
import os
from typing import List
import sys
import datasets

description = "Analizează următoarea recenzie și evalueaz-o ca fiind pozitivă sau negativă.\n"
class LarosedaMultiClass(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"prompt": datasets.Value("string"),
             "response": datasets.Value("string")}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = dl_manager.download("https://raw.githubusercontent.com/ancatache/LaRoSeDa/main/data_splitted/laroseda_train.json")
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data_list = json.load(f)["reviews"]
            for entry_id, entry in enumerate(data_list):
                yield entry_id, {
                    "prompt": "{2}Recenzie: {0}. {1}\nEvaluare:".format(entry["title"].strip(), entry["content"].strip(), description),
                    "response": "pozitivă" if int(entry["starRating"]) > 3 else "negativă"
                }

        