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

from util.log_utils import write_log
from util.parse_time import parseTime

from lib.data_loaders.ClassDatasetSequence import ClassDatasetSequence

from datetime import datetime

SAVE_FOLDER = "../../../../../logs/convclass"
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


def save_viz_improved(
    image: np.ndarray, gt: np.ndarray, sat: np.ndarray, sat_files, inx
):
    prediction = image
    fig, axes = plt.subplots(2, 5)
    for i, a in enumerate(axes[0]):
        a.set_title(parseTime(sat_files[i]))
        a.imshow(sat[i][1])

    axes[-1, 0].set_title("ground truth")
    axes[-1, 0].imshow(gt)
    axes[-1, 1].axis("off")
    axes[-1, 2].axis("off")
    axes[-1, 3].axis("off")
    axes[-1, -1].set_title("prediction")
    axes[-1, -1].imshow(prediction)
    # axes[-1, -1].imshow(image)

    save_path = f"{SAVE_FOLDER}/experiment-{inx}.png"
    fig.savefig(save_path, dpi=300)


@step
def visualize(
    predict_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model: pl.LightningModule,
    file_list: Dict,
) -> np.ndarray:
    trainer = pl.Trainer()
    result = trainer.predict(model, predict_dataloader)

    dataset = ClassDatasetSequence(
        satellite_files=file_list["test"]["sat"],
        radar_files=file_list["test"]["rad"],
        radar_seq_len=1,
    )
    secondDataloader = iter(DataLoader(dataset, batch_size=1, drop_last=True))

    for i, r in enumerate(result):
        write_log(f"{len(r)}")
        for batch in r:
            prediction = batch.view(8, 256, 256).cpu().detach().numpy()
            data, y = next(secondDataloader)
            gt = y.view(256, 256).cpu().detach().numpy()
            sat = data.view(5, 12, 256, 256).cpu().detach().numpy()
            save_viz_improved(prediction, gt, sat, file_list["test"]["sat"], i)

    result = trainer.predict(model, val_dataloader)

    dataset = ClassDatasetSequence(
        satellite_files=file_list["valid"]["sat"],
        radar_files=file_list["valid"]["rad"],
        radar_seq_len=1,
    )
    secondDataloader = iter(DataLoader(dataset, batch_size=1, drop_last=True))

    for i, r in enumerate(result):
        write_log(f"{len(r)}")
        for batch in r:
            prediction = batch.view(8, 256, 256).cpu().detach().numpy()
            data, y = next(secondDataloader)
            gt = y.view(256, 256).cpu().detach().numpy()
            sat = data.view(5, 12, 256, 256).cpu().detach().numpy()
            save_viz_improved(prediction, gt, sat, file_list["valid"]["sat"], i)

    return np.zeros((10, 10))
