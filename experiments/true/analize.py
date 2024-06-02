from scale_score.eval import evaluate_scale
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json



with open("./experiments/true/results/scale_xl_true_results.json", "r") as f:
    data = json.load(f)


print(data.keys())

scores = {}
for k in tqdm(data.keys()):

    res, label = data[k][0], data[k][1]

    metrics = evaluate_scale(res, label)

    roc_auc = roc_auc_score(label, res)

    scores[k] = { "roc_auc": roc_auc, "metrics": metrics}

with open("./experiments/true/scores/scale_xl_true_scores.json", "w") as f:
    json.dump(scores, f)


