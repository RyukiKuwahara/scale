import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from scale_score.scorer import SCALEScorer

files = [
    "begin.csv",
    "dialfact.csv",
    "fever.csv",
    "frank.csv",
    "mnbm.csv",
    "paws.csv",
    "q2.csv",
    "qags_cnndm.csv",
    "qags_xsum.csv",
    "summeval.csv",
    "vitc.csv",
]

data = {}
for f in files:
    data[f.split(".")[0]] = pd.read_csv("../../data/true/" + f)

size = "xl"

scorer = SCALEScorer(size=size, device="cuda")

results = {}
for k in tqdm(data.keys()):
    inf_summ = data[k]["generated_text"].to_numpy()[..., np.newaxis].tolist()
    orig_convo = data[k]["grounding"].to_numpy()[..., np.newaxis].tolist()
    convo = [premiss[0] for premiss in orig_convo]
    retrieval_convo = [premiss[0].split(". ") for premiss in orig_convo]
    label = data[k]["label"].tolist()
    normal_res = scorer.score(convo[:3], inf_summ[:3])
    retrieval_res = scorer.retrieve(retrieval_convo[:3], inf_summ[:3])
    if len(normal_res) != len(retrieval_res):
        raise ValueError(
            "There must be an equal number of normal_res and retrieval_res"
        )
    
    res = []
    mode = []
    for normal_score, retrieval_tuple in zip(normal_res, retrieval_res):
        retrieval_score = retrieval_tuple[1]
        res.append(max(normal_score, retrieval_score))
        if normal_score == retrieval_score:
            mode.append("equal")
        elif normal_score > retrieval_score:
            mode.append("normal win!")
        else:
            mode.append("retrieval win!")
    results[k] = [res, label, mode]
    with open(f"results/scale_{size}_true_retrieval_results.json", "w") as file:
        json.dump(results, file)
