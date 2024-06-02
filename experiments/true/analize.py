from scale_score.eval import evaluate_scale
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
import json



with open("./experiments/true/results/scale_xl_true_results.json", "r") as f:
    data = json.load(f)


print(data.keys())
res, label = data["begin"][0], data["begin"][1]
print(len(res), len(label))

metrics = evaluate_scale(res, label)
print(metrics)

roc_auc = roc_auc_score(label, res)
print("ROC-AUC:", roc_auc)

p_stat = pearsonr(res, label)[0]
print("p_stat", p_stat)


