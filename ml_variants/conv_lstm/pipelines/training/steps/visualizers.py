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


def make_gif(array: np.ndarray):
    # imgs = np.random.randint(0, 255, (100, 50, 50, 3), dtype=np.uint8)
    array = array
    imgs = [Image.fromarray(img) for img in array]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(
        "array.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0
    )


# class GifMaterializer(BaseMaterializer):
#     pass


@step
def visualize(predict_dataloader: DataLoader, model: pl.LightningModule) -> np.ndarray:
    trainer = pl.Trainer()
    result = trainer.predict(model, predict_dataloader)

    print(result[0].size())

    radar_sequence = result[0][0, :, :, :].detach().numpy()

    print(radar_sequence.shape)

    # make_gif(radar_sequence)
    plt.imsave("test_image.png", radar_sequence[0])

    return radar_sequence[0]
