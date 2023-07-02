import lightning.pytorch as pl
import torchmetrics as tm
from typing import Dict, List
import torch


def weight_function(target):
    print(target.shape)
    result = target * 255
    print(result.shape)
    result[result < 2] = 1
    result[(result < 5) & (result >= 2)] = 2
    result[(result < 5) & (result >= 5)] = 5
    result[(result < 30) & (result >= 10)] = 10
    result[result >= 30] = 30
    return result


def rmse():
    return tm.MeanSquaredError(squared=False)


class BMAE(tm.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    sum_squared_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        squared: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        sum_squared_error = torch.mean(
            torch.abs(preds - target) * weight_function(target)
        )
        self.sum_squared_error += sum_squared_error
        self.total += 1

    def compute(self) -> torch.Tensor:
        """Computes mean squared error over state."""
        return self.sum_squared_error / self.total


class BMSE(tm.Metric):
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    sum_squared_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        squared: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.squared = squared

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        sum_squared_error = torch.mean(
            torch.square(preds - target) * weight_function(target)
        )
        self.sum_squared_error += sum_squared_error
        self.total += 1

    def compute(self) -> torch.Tensor:
        """Computes mean squared error over state."""
        return self.sum_squared_error / self.total


metrics_database: Dict[str, tm.Metric] = [
    {"name": "MAE", "metric": tm.MeanAbsoluteError},
    {"name": "MSE", "metric": tm.MeanSquaredError},
    {"name": "RMSE", "metric": rmse},
    {"name": "BMAE", "metric": BMAE},
    {"name": "BMSE", "metric": BMSE},
]


class RegressionMetrics:
    def __init__(self):
        self.metrics = [
            {
                "name": x["name"],
                "metric": x["metric"](),
            }
            for x in metrics_database
        ]

    def eval(self, model: pl.LightningModule, yhat, y, step_name, device):
        for metric in self.metrics:
            m = metric["metric"].to(device)
            result = m(yhat, y)
            model.log(
                f"{step_name}_{metric['name']}",
                result,
                on_epoch=True,
                prog_bar=True,
            )
