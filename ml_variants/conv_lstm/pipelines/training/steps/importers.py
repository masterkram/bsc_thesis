from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from zenml.steps import Output, step
from satpy import Scene
import numpy as np
import sys

sys.path.append("../")
sys.path.append("../../../../")
sys.path.append("../lib/")
sys.path.append("./lib/")


from lib.DatasetDistributor import DatasetDistributor
from lib.DatasetType import DatasetType
from lib.Sat2RadModule import Sat2RadDataModule

PATH_TO_DATA = "../../../../data"


@step(enable_cache=False)
def importer_sat2rad() -> (
    Output(
        train_dataloader=DataLoader,
        val_dataloader=DataLoader,
        test_dataloader=DataLoader,
        predict_dataloader=DataLoader,
    )
):
    data_module = Sat2RadDataModule(data_dir=PATH_TO_DATA, batch_size=1)
    data_module.prepare_data()
    data_module.setup(None)
    return (
        data_module.train_dataloader(),
        data_module.val_dataloader(),
        data_module.test_dataloader(),
        data_module.predict_dataloader(),
    )


if __name__ == "__main__":
    data_module = Sat2RadDataModule(data_dir="../../../../../data", batch_size=1)
    data_module.prepare_data()
    data_module.setup(None)
    dataset = iter(data_module.train_dataloader())
    # print(next(dataset)[1].size())
    mygt = [x[1].detach().numpy() for x in iter(dataset)]
    print(len(mygt))
    print(mygt[0][0].shape)
