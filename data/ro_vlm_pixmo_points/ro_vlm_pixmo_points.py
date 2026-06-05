import json
import os
from typing import List
import sys
import datasets
import random
random.seed(42)

count_prompts = [
"Indicați spre {label}\nVă rugăm să spuneți «Nu există în imagine.» dacă nu există.",
"Indicați toate aparițiile lui \"{label}\"",
"Indicați spre orice {label} din imagine.",
"Indicați: Unde sunt {label}",
"Arată-mi unde sunt {label}",
"Poți să-mi arăți unde sunt {label}?",
"Arată-mi unde sunt {label}",
"Arată-mi unde este un {label}",
"Arată-mi unde este o {label}",
"Există {label} în imagine? Arată-mi unde sunt.",
"Unde sunt {label}?",
"Generează o listă de puncte care arată unde sunt {label}.",
"Găsește \"{label}\".",
"Găsește un \"{label}\".",
"Localizează toate {label}.",
"Localizează un {label}.",
"Localizează o {label}.",
"Localizează fiecare {label}.",
"Localizează {label}.",
"Obiect: {label}\nInstrucțiune: Indică obiectul.",
"găsește {label}",
"găsește {label}.",
"Indică fiecare {label}",
"găsește orice {label} în imagine",
"Găsește {label}",
"Găsește orice {label}",
"Indică un {label}",
"Indică o {label}",
"Caută {label} în imagine și arată-mi unde sunt.",
"Ajută-mă să găsesc un obiect în imagine indicându-l.\nObiect: {label}.",
"Caut {label}, unde pot fi găsite în imagine?",
"Poți vedea vreun {label} în imagine? Indică-le.",
"Indică fiecare {label} în imagine.",
"Indică fiecare {label} în imagine.",
"Indică {label} în imagine.",
"Localizează fiecare {label} în imagine.",
"Poți indica toate {label} din această imagine?",
"Te rog să găsești {label} și să-mi arăți unde sunt.",
"Dacă există {label} prezente, indică pozițiile lor.",
"Dacă există un {label} prezent, indică pozițiile sale.",
"Arată-mi toate {label} vizibile",
]


class PixmoPoints(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self):
        features = datasets.Features(
            {"conversations": [{"from": datasets.Value("string"), "value": datasets.Value("string")}],
            "images": [datasets.Value("string")]}
        )
        return datasets.DatasetInfo(features=features)

    def _split_generators(self, dl_manager):
        file_path = "data/ro_vlm_pixmo_points_sampled_0.12_10.json"
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": file_path})]

    def _generate_examples(self, filepath: str):
        data = json.load(open(filepath, "r", encoding="utf-8"))
        for entry_id, entry in enumerate(data):
            new_entry = {}
            new_entry["conversations"] = []
            prompt = random.choice(count_prompts)
            if random.random() < 0.25:
                prompt = prompt + "\nRăspundeți cu o listă de puncte în forma \"x1\" \"y1\" \n\"x2\" \"y2\" \n... sau cu «Nu există în imagine.» dacă nu există niciunul."
            if random.random() < 0.5:
                new_entry["conversations"].append({"from": "human", "value": "<image>\n" + prompt.format(label=entry["label"])})
            else:
                new_entry["conversations"].append({"from": "human", "value": prompt.format(label=entry["label"]) + "\n<image>"})

            if len(entry["points"]) == 0:
                if random.random() < 0.5:
                    formatted_out = "Nu există {0} în imagine.".format(entry["label"])
                else:
                    formatted_out = "Nu există în imagine.".format(entry["label"])
            else:
                # formatted_out = "<points<|points|> alt=\"{label}\">{label}</points>".format(label=entry["label"])
                # si = 1
                # s = ""
                # for point in entry["points"][:20]:
                #     s += " x{0}=\"{1}\"".format(si, point["x"])
                #     s += " y{0}=\"{1}\"".format(si, point["y"])
                #     si += 1
                # formatted_out = formatted_out.replace("<|points|>", s)
                s = ""
                for point in entry["points"]:
                    s += "\"{0:.2f}\" \"{1:.2f}\" \n".format(point["x"], point["y"])                    
                formatted_out = s.strip()
            new_entry["conversations"].append({"from": "gpt", "value": formatted_out})
            new_entry["images"] = [os.path.join("/export/projects/nlp/datasets/rovlm/", entry["images"][0])]
            yield entry_id, new_entry


if __name__ == "__main__":

    aya = PixmoPoints()
    x = aya._generate_examples("data/ro_vlm_pixmo_points_sampled_0.12_10.json")
    for i in range(5):
        print(next(x))
        print(next(x))