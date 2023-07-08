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
from util.parse_time import parseTime
from lib.data_loaders.ClassDatasetSequence import ClassDatasetSequence
from util.log_utils import write_log

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


def save_viz_improved(
    image: np.ndarray, gt: np.ndarray, sat: np.ndarray, sat_files, inx
):
    prediction = np.argmax(image, axis=0)
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
    predict_dataloader: DataLoader, model: pl.LightningModule, file_list: Dict
) -> np.ndarray:
    trainer = pl.Trainer()
    result = trainer.predict(model, predict_dataloader)

    # sample_output = 0

    # gt_files = file_list["test"]["rad"]
    # print(result[0].size())
    # example_radar_sequence = result[0][sample_output].view(256, 256).detach().numpy()

    # ground_truth = [np.load(x) for x in gt_files[0:1]]
    # example_radar_sequence_gt = ground_truth[sample_output]
    # active_experiment_name = "convlsm-predicting-classes"
    # save_viz(example_radar_sequence, active_experiment_name)
    # save_viz(example_radar_sequence_gt, active_experiment_name, False)

    # all_predictions = [x[0].view(256, 256).detach().numpy() for x in result]
    # make_gif(all_predictions)
    # make_gif(ground_truth)
    dataset = ClassDatasetSequence(
        satellite_files=file_list["test"]["sat"],
        radar_files=file_list["test"]["rad"],
        radar_seq_len=1,
    )
    secondDataloader = iter(DataLoader(dataset, batch_size=1, drop_last=True))

    for i, r in enumerate(result):
        write_log(f"{len(r)}")
        for batch in r:
            write_log(f"in a batch {len(r)}")
            write_log(f"in a batch torch ? {r.shape}")

            prediction = batch.view(8, 256, 256).cpu().detach().numpy()
            data, y = next(secondDataloader)
            # print(y.shape)
            # print(data.shape)
            gt = y.view(256, 256).cpu().detach().numpy()
            sat = data.view(5, 12, 256, 256).cpu().detach().numpy()
            save_viz_improved(prediction, gt, sat, file_list["test"]["sat"], i)

    return np.zeros((10, 10))
