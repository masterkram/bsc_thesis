from torch.utils.data import DataLoader
import lightning.pytorch as pl

from zenml.steps import step


@step
def evaluator(test_data: DataLoader, model: pl.LightningModule) -> float:
    """
    Evaluates the `model`

    Parameters:
    + test_dataloader (`DataLoader`): DataLoader containing the test partition.
    + model (`LightningModule`): The trained model.

    Returns:
    + acc (`float`):The accuracy of the model on the test partition.
    """

    trainer = pl.Trainer()

    result = trainer.test(model, test_data)

    return result[0]["acc"]
