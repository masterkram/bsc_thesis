from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from enum import Enum
from zenml.steps import Output, step
import lightning.pytorch as pl
import numpy as np
from zenml.materializers.base_materializer import BaseMaterializer
from PIL import Image
import matplotlib.pyplot as plt
import mlflow

SAVE_FOLDER = "./viz/"
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


def save_image_grid(images: np.ndarray, active_experiment_name: str, pred: bool = True):
    fig, axs = plt.subplots(3, 4, figsize=(10, 8))
    # fig.tight_layout()

    # Loop through the images and display them
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i], cmap=CMAP)  # Display each image in grayscale
        ax.axis("off")
        ax.set_title(f"Image t = {(i+1) * 15}min")

    variant_string = "ConvLSTM\nSat2Rad-" if pred else "Ground Truth Radar Images"
    title = f"{variant_string}({active_experiment_name})"
    fig.suptitle(
        f"ConvLSTM\nSat2Rad-({active_experiment_name})",
        fontsize=16,
        fontweight="bold",
    )
    # Show the plot
    save_type = "pred" if pred else "gt"
    save_path = f"{SAVE_FOLDER}experiment-{active_experiment_name}-{save_type}.png"
    plt.savefig(save_path, dpi=300)


@step
def visualize(predict_dataloader: DataLoader, model: pl.LightningModule) -> np.ndarray:
    trainer = pl.Trainer()
    result = trainer.predict(model, predict_dataloader)

    radar_sequence = result[0][0, :, :, :].detach().numpy()

    gt = [x[1].detach().numpy()[0] for x in iter(predict_dataloader)]

    active_experiment_name = "dummy_name"
    save_image_grid(radar_sequence, active_experiment_name)
    save_image_grid(gt, active_experiment_name, pred=False)
    make_gif(images)
    make_gif(gt, pred=False)
    # mlflow.log_artifact("test_image.png")

    return radar_sequence[0]


if __name__ == "__main__":
    print("welcome to viz")
    np.random.seed(42)
    images = np.random.rand(12, 64, 64)
    active_experiment_name = "ba"
    save_image_grid(images, active_experiment_name)
    save_image_grid(images, active_experiment_name, pred=False)
    make_gif(images)
    make_gif(images)
