import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from scale_score.scorer import SCALEScorer

with open("../../data/screen_eval.json", "r") as file:
    data = json.load(file)

fetch = SCALEScorer(size="xl")

rank_list = []
scores = []
# Get index and replace with '' to maintian correct idx without duplicate matching
convo = [
    data["original_convo"][k].split("\n")
    for idx, k in enumerate(data["original_convo"].keys())
    if data["agg_label"][f"{idx}"]
]
convo_for_mod = [
    data["original_convo"][k].split("\n")
    for idx, k in enumerate(data["original_convo"].keys())
    if data["agg_label"][f"{idx}"]
]
summaries = [
    data["inferred_summary"][k]
    for idx, k in enumerate(data["original_convo"].keys())
    if data["agg_label"][f"{idx}"]
]
rel_utts = [
    data["rel_utt"][k]
    for idx, k in enumerate(data["rel_utt"].keys())
    if data["agg_label"][f"{idx}"]
]
assert len(rel_utts) == len(convo)
assert len(summaries) == len(convo)

res = []
for i in tqdm(range(len(convo))):
    rank_list = []
    scores = []
    summary = summaries[i]
    summary = ["Krueger et al. propose Bayesian hypernetworks, which transform a simple noise distribution to approximate the posterior distribution over the parameters of a primary network."] #target
    convo = ["We study Bayesian hypernetworks: a framework for approximate Bayesian inference in neural networks. A Bayesian hypernetwork h is a neural network which learns to transform a simple noise distribution, p(ε) = N(0, I), to a distribution q(θ) := q(h(ε)) over the parameters θ of another neural network (the \"primary network\"). We train q with variational inference, using an invertible h to enable efficient estimation of the variational lower bound on the posterior p(θ|D) via sampling. In contrast to most methods for Bayesian deep learning, Bayesian hypernetworks can represent a complex multimodal approximate posterior with correlations between parameters, while enabling cheap iid sampling of q(θ). In practice, Bayesian hypernetworks can provide a better defense against adversarial examples than dropout, and also exhibit competitive performance on a suite of tasks which evaluate model uncertainty, including regularization, active learning, and anomaly detection."] #source
    convo = convo[0].split(". ")

    t0 = time.time()
    results = fetch.retrieve([convo[i]], [summary], branches=2)
    t1 = time.time()
    print(results)
    exit(1)
    rel_utt_idx = convo_for_mod[i].index(results[0][0])
    rank_list.append(rel_utt_idx)
    scores.append(results[0][1])

    r1 = int(rank_list[0] in rel_utts[i])
    retrieval = [r1]
    res.append(
        {
            "rank_list": rank_list,
            "scores": scores,
            "rel_utts": rel_utts[i],
            "retrieval": retrieval,
            "time": t1 - t0,
        }
    )
    with open("results/scale_xl_retrieval.json", "w") as file:
        json.dump(res, file)
