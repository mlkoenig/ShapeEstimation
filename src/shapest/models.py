"""
Module for the implementation of the PyTorch and CAESAR prediction models.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


class CaesarModel:
    def __init__(self, root: Path, k: int = 10, sigma: float = 3.0) -> None:
        """A class to generate the vertex positions of the standardized mesh topology
        for eigenvalue coefficients.

        Args:
            root (Path): The path of the caesar files:
                (`meanShape.mat`, `evalues.mat`, `evectors.mat`).
            k (int): The number of principal components to use. Defaults to 10.
            sigma (float): The range of the shape space. Defaults to 3.0.
        """
        super().__init__()

        self.k = k
        self.sigma = sigma
        self.mean = scipy.io.loadmat(root / "meanShape.mat")["points"]
        self.evals = scipy.io.loadmat(root / "evalues.mat")["evalues"][:, :k]
        self.evals = np.sqrt(self.evals)
        self.evecs = scipy.io.loadmat(root / "evectors.mat")["evectors"][:k, :]

        data = pd.read_table(
            root / "model.dat", delim_whitespace=True, decimal=".", header=0, skiprows=1
        )[["x", "y", "z"]]

        self.points = data.iloc[0:6449]
        self.header = ["6449", "12894", "0"]
        self.faces = data.iloc[6449:19343]

    def __call__(
        self, coeffs: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the vertex positions of the shape given by a coefficient vector.

        If `coeffs` is None, a random vector will be used.

        The formula for calculating new positions P in the shape space interpolation is

        P = U * phi + M

        where U is the matrix of eigen vectors, M the dataset mean of the vertex
        positions and phi the interpolated coefficient vector, which is interpolated
        in the eigen value range with range sigma.

        Args:
            coeffs (Optional[np.ndarray], optional): The coefficient vector of the
                shape. This needs have `k` coefficients in the range of [0, 1].
                Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: The coefficient vector and the position
                matrix of the standardized mesh for `coeffs`.
        """
        if coeffs is None:
            coeffs = npr.rand(self.k)
        phi = -self.sigma * self.evals + (self.sigma + 1.0) * coeffs * self.evals
        positions = phi.dot(self.evecs).reshape((-1, 3), order="F") + self.mean
        return coeffs, positions

    def to_off(self, filename: Path, positions: np.ndarray) -> None:
        """
        Saves a mesh file in OFF format for the given vertex positions, i.e. the
        vertex positions from calling the model.

        Args:
            filename (Path): Where to save the OFF file.
            positions (np.ndarray): The vertex positions.

        Raises:
            ValueError: If the filename is not in OFF format.
        """
        if not filename.suffix == ".off":
            raise ValueError(
                f"Invalid filename extension {filename.suffix}. Use '.off' instead."
            )

        with open(filename, "w") as f:
            f.write("OFF\n\n")
            f.write(" ".join(self.header) + "\n")

            for p in positions:
                f.write(" ".join(map(str, p)))
                f.write("\n")

            for _, fa in self.faces.iterrows():
                f.write(str(len(fa)) + " ")
                f.write(" ".join(map(str, map(int, fa))))
                f.write("\n")


class Encoder(nn.Module):
    def __init__(self):
        """
        PyTorch backbone for a simple encoder CNN model.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.fc1 = nn.Linear(239616, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


class ShapeModel(LightningModule):
    def __init__(self, model: nn.Module, inv_transform=None):
        """
        Lightning module to train a backbone model on the human shape estimation task.

        Args:
            model (nn.Module): The backbone PyTorch model.
            inv_transform (_type_, optional): An inverse transformation for model
                predictions. Defaults to None.
        """
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
