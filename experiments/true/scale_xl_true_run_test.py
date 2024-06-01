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
    # inf_summ = data[k]["generated_text"].to_numpy()[..., np.newaxis].tolist()[0]
    # convo = data[k]["grounding"].to_numpy()[..., np.newaxis].tolist()[0]
    inf_summ = ["he served as the president from 1946."] #target
    convo = ["Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who served as the 45th president of the United States from 2017 to 2021."] #source
    print("inf_summ: ", inf_summ)
    print("convo: ", convo)
    label = data[k]["label"].tolist()[0]
    res = scorer.score(convo, [inf_summ])
    results[k] = [res, label]
    print(results)
    break
