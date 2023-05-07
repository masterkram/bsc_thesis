import torch
from torch import nn
from torch.utils.data import DataLoader

from zenml.steps import step

# Get cpu or gpu device for training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@step
def evaluator(test_dataloader: DataLoader, model: nn.Module) -> float:
    """Evaluates on the model."""
    loss_fn = nn.CrossEntropyLoss()

    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return 100 * correct
