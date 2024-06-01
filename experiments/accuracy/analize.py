from scale_score.eval import evaluate_scale
from sklearn.metrics import roc_auc_score

with open("./experiments/accuracy/results/scale_xxl_results.json", "r") as f:
    lines = f.readlines()

lines = eval(lines[0])
print(len(lines), type(lines))
res, label = lines[0], lines[1]
print(len(res), len(label))

metrics = evaluate_scale(res, label)
print(metrics)

roc_auc = roc_auc_score(label, res)
print("ROC-AUC:", roc_auc)


