from pathlib import Path
from typing import Any

import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class CaesarModel:
    def __init__(self, root: Path) -> None:
        self.mean = scipy.io.loadmat(root / "meanShape.mat")["points"]
        self.evals = scipy.io.loadmat(root / "evalues.mat")["evalues"]
        self.evals = np.sqrt(self.evals)
        self.evecs = scipy.io.loadmat(root / "evectors.mat")["evectors"]

        data = pd.read_table(
            root / "model.dat", delim_whitespace=True, decimal=".", header=0, skiprows=1
        )[["x", "y", "z"]]

        self.points = data.iloc[0:6449]
        self.header = ["6449", "12894", "0"]
        self.faces = data.iloc[6449:19343]

    def random_phi(self, k=10, sigma=3.0):
        eigen_vals = self.evals[:, :k]
        return -sigma * eigen_vals + (sigma + 1.0) * npr.rand(k) * eigen_vals

    def predict(self, phi=None, k=10, sigma=3.0):
        if phi is None:
            phi = self.random_phi(k, sigma)

        return phi.dot(self.evecs[:k, :]).reshape((-1, 3), order="F") + self.mean

    def to_off(self, filename: Path, position: np.ndarray):
        if not filename.suffix == ".off":
            raise ValueError(
                f"Invalid filename extension {filename.suffix}. Use '.off' instead."
            )

        with open(filename, "w") as f:
            f.write("OFF\n\n")
            f.write(" ".join(self.header) + "\n")

            for p in position:
                f.write(" ".join(map(str, p)))
                f.write("\n")

            for _, fa in self.faces.iterrows():
                f.write(str(len(fa)) + " ")
                f.write(" ".join(map(str, map(int, fa))))
                f.write("\n")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(239616, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv4(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.selu(self.fc1(x))
        x = self.fc2(x)
        return x


class ShapeModel(LightningModule):
    def __init__(self, model: nn.Module, inv_transform=None):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "inv_transform"])

        self.model = model
        self.inv_transform = inv_transform
        self.criterion = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx) -> Any:
        x, _ = batch

        y_pred = self(x)

        if self.inv_transform is not None:
            return self.inv_transform(y_pred)

        return y_pred

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        y_hat = self.model(x)
        self.log(f"{prefix}_loss", self.criterion(y_hat, y))
        self.log(f"{prefix}_mae", self.mae(y_hat, y))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)
