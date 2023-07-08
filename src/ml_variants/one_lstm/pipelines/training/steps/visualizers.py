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
from datetime import datetime
from util.parse_time import find_matching_string, parseTime

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


def save_viz(image: np.ndarray, gt: np.ndarray, sat: np.ndarray, sat_files):
    now = str(datetime.now())

    fig, axes = plt.subplots(2, 5)
    for i, a in enumerate(axes[0]):
        a.set_title(parseTime(sat_files[i]))
        a.imshow(sat[i][0])
    # for i, a in enumerate(axes[1]):
    #     a.set_title("timestamp")
    #     a.imshow(radar[i])
    axes[-1, 0].set_title("ground truth")
    axes[-1, 0].imshow(gt)
    axes[-1, 1].axis("off")
    axes[-1, 2].axis("off")
    axes[-1, 3].axis("off")
    axes[-1, -1].set_title("prediction")
    axes[-1, -1].imshow(image)

    save_path = f"{SAVE_FOLDER}experiment-{now}.png"
    fig.savefig(save_path, dpi=300)


@step
def visualize(
    predict_dataloader: DataLoader, model: pl.LightningModule, file_list: Dict
) -> np.ndarray:
    # trainer = pl.Trainer()
    # result = trainer.predict(model, predict_dataloader)
    # sample_output = 0

    # gt_files = file_list["test"]["rad"][0:5]

    # example_radar_sequence = (
    # result[0][sample_output, :, :].view(256, 256).detach().numpy()
    # )

    # ground_truth = [np.reshape(np.load(x), (256, 256)) for x in gt_files]
    # example_radar_sequence_gt = ground_truth[sample_output]
    # save_viz(example_radar_sequence, active_experiment_name)
    # save_viz(example_radar_sequence_gt, active_experiment_name, False)
    # make_gif(all_predictions)
    # make_gif(ground_truth)
    # save_viz(
    #     example_radar_sequence,
    #     ground_truth,
    # )
    sat_files = file_list["test"]["sat"][0:5]
    sat = np.array([np.load(x) for x in sat_files])
    satellite_files = torch.from_numpy(sat).float().view(1, 5, 11, 256, 256).cuda()
    ground_truth_ix = find_matching_string(file_list["test"]["rad"], sat_files[-1])
    ground_truth_np = np.load(file_list["test"]["rad"][ground_truth_ix])
    ground_truth = torch.from_numpy(ground_truth_np)
    print(satellite_files.shape)

    result = (
        model((satellite_files, ground_truth)).view(256, 256).detach().cpu().numpy()
    )

    save_viz(
        result,
        ground_truth_np,
        sat,
        sat_files,
    )

    return result


if __name__ == "__main__":
    now = str(datetime.now())
    prediction = np.random.randint(0, 10, (10, 10))
    gt = np.random.randint(0, 10, (10, 10))
    sat = np.random.randint(0, 10, (5, 10, 10))
    radar = np.random.randint(0, 10, (5, 10, 10))

    fig, axes = plt.subplots(3, 5)
    for i, a in enumerate(axes[0]):
        a.set_title("timestamp")
        a.imshow(sat[i])
    for i, a in enumerate(axes[1]):
        a.set_title("timestamp")
        a.imshow(radar[i])
    axes[-1, 0].set_title("ground truth")
    axes[-1, 0].imshow(gt)
    axes[-1, 1].axis("off")
    axes[-1, 2].axis("off")
    axes[-1, 3].axis("off")
    axes[-1, -1].set_title("prediction")
    axes[-1, -1].imshow(prediction)

    save_path = f"{SAVE_FOLDER}experiment-{now}.png"
    plt.savefig(save_path, dpi=300)
