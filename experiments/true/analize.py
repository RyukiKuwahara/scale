from scale_score.eval import evaluate_scale
from sklearn.metrics import roc_auc_score
import json

def chank(array: list, w: int = 10000):
    chunks = [array[i:i+w] for i in range(0, len(array), w)]
    return chunks

with open("./experiments/true/results/scale_large_true_results.json", "r") as f:
    data = json.load(f)


print(data.keys())
res, label = data["scores"], data["labels"]
print(len(res), len(label))

ress = [chank(res)[0]]
labels = [chank(label)[0]]
print(len(ress), len(labels))

for res, label in zip(ress, labels):
    metrics = evaluate_scale(res, label)
    print(metrics)

    roc_auc = roc_auc_score(label, res)
    print("ROC-AUC:", roc_auc)


