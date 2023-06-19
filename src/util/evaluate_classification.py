import lightning.pytorch as pl
import torchmetrics as tm
from typing import Dict, List

metrics_database: Dict[str, tm.Metric] = [
    {"name": "acc", "metric": tm.Accuracy},
    {"name": "precision", "metric": tm.Precision},
    {"name": "recall", "metric": tm.Recall},
    {"name": "exact", "metric": tm.ExactMatch},
    {"name": "f1", "metric": tm.F1Score},
    {"name": "jaccard", "metric": tm.JaccardIndex},
]


class ClassifierMetrics:
    def __init__(self, metrics: List, num_classes=8, task="multiclass"):
        self.metrics = [
            {
                "name": x["name"],
                "metric": x["metric"](num_classes=num_classes, task=task),
            }
            for x in metrics_database
            if x["name"] in metrics
        ]

    def evaluate_classification(
        self, model: pl.LightningModule, yhat, y, step_name, device
    ):
        for metric in self.metrics:
            m = metric["metric"].to(device)
            result = m(yhat, y)
            model.log(
                f"{step_name}_{metric['name']}",
                result,
                on_epoch=True,
                prog_bar=True,
            )
