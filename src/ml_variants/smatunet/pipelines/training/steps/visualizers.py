from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from enum import Enum
from zenml.steps import Output
from zenml import step
import lightning.pytorch as pl
import numpy as np
from zenml.materializers.base_materializer import BaseMaterializer
from PIL import Image
import matplotlib.pyplot as plt
import mlflow
import math
from typing import Dict

SAVE_FOLDER = "../../../../../logs/"
CMAP = "Blues"


def make_gif(array: np.ndarray, pred=True):
    # imgs = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)
    array = array * 255
    imgs = [Image.fromarray(img) for img in array]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    file_name = "pred" if pred else "gt"
    imgs[0].save(
        f"{SAVE_FOLDER}{file_name}_vid.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=50,
        loop=0,
    )


# class GifMaterializer(BaseMaterializer):
#     pass


def save_viz(image: np.ndarray, active_experiment_name: str, pred: bool = True):
    plt.imshow(image)
    # save the plot
    save_type = "pred" if pred else "gt"
    save_path = f"{SAVE_FOLDER}experiment-{active_experiment_name}-{save_type}.png"
    plt.savefig(save_path, dpi=300)


@step
def visualize(
    predict_dataloader: DataLoader, model: pl.LightningModule, file_list: Dict
) -> np.ndarray:
    trainer = pl.Trainer()
    result = trainer.predict(model, predict_dataloader)

    sample_output = 0

    gt_files = file_list["test"]["rad"]
    print(len(result))
    print(result[0].size())
    example_radar_sequence = result[0].reshape(256, 256).detach().numpy()

    all_predictions = [x.detach().numpy() for x in result]

    ground_truth = [np.load(x) for x in gt_files]
    example_radar_sequence_gt = ground_truth[sample_output]
    active_experiment_name = "dummy_name"
    save_viz(example_radar_sequence, active_experiment_name)
    save_viz(example_radar_sequence_gt, active_experiment_name, False)

    make_gif(all_predictions)
    make_gif(ground_truth)

    return example_radar_sequence


if __name__ == "__main__":
    save_viz(np.random.random((256, 256)), "rerere")
