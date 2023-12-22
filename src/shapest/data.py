"""
Module for data-related implementations.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import pyvista as pv
import torch
from lightning.pytorch import LightningDataModule, seed_everything
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from tqdm import tqdm

from .models import CaesarModel

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0], std=[255.0, 255.0]),
    ]
)


class Scaler(torch.nn.Module):
    """
    A point-wise scaling transformation.
    """

    def __init__(self, scl: float) -> None:
        super().__init__()
        self.scl = scl

    def forward(self, x):
        return x * self.scl


def plot_mesh(
    pl: pv.Plotter,
    filename: Path,
    cam="xz",
    xy_pos=(-3600, -3600),
) -> pv.Plotter:
    """
    Add a mesh file to the plotter and set up the camera to normalized coordinates.


    Args:
        pl (pv.Plotter): The plotter instance.
        filename (Path): The mesh file (OFF format).
        cam (str, optional): The camera view. Either "xz" for for side-view or "yz"
            for front-view Defaults to "xz".
        xy_pos (tuple, optional): The camera translation in x and y direction.
            Defaults to (-3600, -3600).

    Returns:
        pv.Plotter: The plotter for chaining.
    """
    mesh = pv.read(filename)
    center = np.array(mesh.center)

    mesh.translate(-center, inplace=True)
    mesh.translate((0, 0, -mesh.bounds[4] - 850), inplace=True)
    c_pos = np.array((0, xy_pos[1], 0)) if cam == "yz" else np.array((xy_pos[0], 0, 0))

    pl.clear()
    pl.add_mesh(mesh, color="white", smooth_shading=True)
    pl.set_background("black")
    pl.camera.position = c_pos
    pl.camera.focal_point = (0, 0, 0)

    return pl


def random_sample(
    pl: pv.Plotter, target_file: Path, caesar_model: CaesarModel, off_file: Path
):
    """
    Randomly samples a new image, target pair and saves them to the target file.

    Args:
        pl (pv.Plotter): The plotter instance.
        target_file (Path): The target filename.
        caesar_model (CaesarModel): The caesar model to obtain the mesh file.
        off_file (Path): A path to save the intermediate mesh file.
    """
    coeffs, positions = caesar_model()
    caesar_model.to_off(off_file, positions)

    xz = plot_mesh(pl, off_file, cam="xz").screenshot()
    yz = plot_mesh(pl, off_file, cam="yz").screenshot()

    # RGB to grayscale conversion
    xz = np.dot(xz, [0.299, 0.587, 0.114]) / 255.0
    yz = np.dot(yz, [0.299, 0.587, 0.114]) / 255.0

    img = np.stack([xz, yz], axis=0)
    torch.save(
        (img.astype(np.float32), np.squeeze(coeffs).astype(np.float32)), target_file
    )


def generate_dataset(
    dataset: str,
    n_samples: int,
    k: int = 10,
    sigma: float = 3.0,
    path: Optional[Path] = None,
):
    """
    Generates a dataset of `n_samples` samples.

    Args:
        dataset (str): The caesar dataset path.
        n_samples (int): The number of samples in the dataset.
        k (int): The number of principal components.
        sigma (float): The variance range for sampling.
        path (Optional[Path], optional): The new dataset path.
            If no path is provided, the files are saved under `datasets` in cwd.
            Defaults to None.
    """
    path = path or Path.cwd() / "datasets"

    path.mkdir(exist_ok=True)
    (path / "train").mkdir(exist_ok=True)
    (path / "test").mkdir(exist_ok=True)

    caesar_model = CaesarModel(Path(dataset), k=k, sigma=sigma)

    pl = pv.Plotter(window_size=(270, 480), off_screen=True)

    for i in tqdm(range(int(n_samples))):
        split = int(n_samples * 0.7)
        target_dataset = "train" if i < split else "test"
        target_file = path / target_dataset / f"sample{i % split}.pt"

        if not target_file.exists():
            if target_dataset == "train":
                with TemporaryDirectory() as tmpdir:
                    off_file = Path(tmpdir) / "tmp.off"
                    random_sample(pl, target_file, caesar_model, off_file)
            else:
                off_file = target_file.with_suffix(".off")
                random_sample(pl, target_file, caesar_model, off_file)

    pl.close()


class HumanImageDataset(Dataset):
    def __init__(self, img_dir: Path, transform=None, target_transform=None):
        """
        Implementation of an image dataset in PyTorch.

        Args:
            img_dir (Path): The dataset path.
            transform (Any): Input transformation. Defaults to None.
            target_transform (Any): Target transformation. Defaults to None.
        """
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.target_transform = target_transform

        self.files = list(self.img_dir.rglob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image, target = torch.load(self.files[idx])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target


class DataModule(LightningDataModule):
    def __init__(
        self,
        path: Path,
        caesar_dataset: Path,
        transform=None,
        target_transform=None,
        batch_size: int = 128,
        shuffle: bool = True,
        split: float = 0.25,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        """
        A lightning data module to generate the dataloaders for model training.

        Args:
            path (Path): Path to the generated dataset root.
            transform (Any, optional): Input transformation. Defaults to None.
            target_transform (Any, optional): Target transformation. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 128.
            shuffle (bool, optional): Whether to shuffle in training. Defaults to True.
            split (float, optional): Validation split. Defaults to 0.25.
            num_workers (int, optional): Number of workers for loading. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory. Defaults to True.
            seed (int, optional): The seed for seed_everything. Defaults to 42.
        """
        super().__init__()

        self.save_hyperparameters(ignore=["target_transform"])

        self.path = Path(path)
        self.caesar = caesar_dataset
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        generate_dataset(self.caesar, 1000, k=10, sigma=3.0, path=self.path)

    def setup(self, stage: str) -> None:
        seed_everything(self.seed)

        if stage == "fit":
            dataset = HumanImageDataset(
                self.path / "train", self.transform, self.target_transform
            )

            self.train_dataset, self.val_dataset = random_split(
                dataset, (1.0 - self.split, self.split)
            )

        elif stage == "test":
            self.test_dataset = HumanImageDataset(
                self.path / "test", self.transform, self.target_transform
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, self.batch_size)
