from pathlib import Path

import numpy as np
import numpy.random as npr
import pandas as pd
import pyvista as pv
import scipy
from PIL import Image

cam_xz = np.array([0, -3600, 0])
cam_yz = np.array([-3600, 0, 0])
window_size = (270, 480)


def load_image(filename: str):
    img = Image.open(filename).convert("L")
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def plot_mesh(filename: Path, cam="xz") -> pv.Plotter:
    mesh = pv.read(filename)
    center = np.array(mesh.center)

    mesh.translate(-center, inplace=True)
    mesh.translate((0, 0, -mesh.bounds[4] - 850), inplace=True)

    pl = pv.Plotter(window_size=window_size)
    pl.add_mesh(mesh, color="white", smooth_shading=True)
    pl.set_background("black")
    pl.camera.position = cam_yz if cam == "yz" else cam_xz
    pl.camera.focal_point = (0, 0, 0)

    return pl


class CaesarModel:
    def __init__(self, root: Path) -> None:
        self.mean = scipy.io.loadmat(root / "meanShape.mat")["points"]
        self.evals = scipy.io.loadmat(root / "evalues.mat")["evalues"]
        self.evars = np.sqrt(self.evals)
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
