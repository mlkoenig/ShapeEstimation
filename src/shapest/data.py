from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import pyvista as pv
import torch
from lightning.pytorch import LightningDataModule, seed_everything
from lightning.pytorch.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
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
    def __init__(self, scl: float) -> None:
        super().__init__()
        self.scl = scl

    def forward(self, x):
        return x * self.scl


target_transform = Scaler(0.001)
inv_transform = Scaler(1000.0)


def plot_mesh(
    pl: pv.Plotter,
    filename: Path,
    cam="xz",
    xy_pos=(-3600, -3600),
) -> pv.Plotter:
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
    pl: pv.Plotter, target_file: Path, caesar_model: CaesarModel, k: int, off_file: Path
):
    phi = caesar_model.random_phi(k)
    position = caesar_model.predict(phi, k=k)

    caesar_model.to_off(off_file, position)

    xz = plot_mesh(pl, off_file, cam="xz").screenshot()
    yz = plot_mesh(pl, off_file, cam="yz").screenshot()

    # RGB to grayscale conversion
    xz = np.dot(xz, [0.299, 0.587, 0.114]) / 255.0
    yz = np.dot(yz, [0.299, 0.587, 0.114]) / 255.0

    img = np.stack([xz, yz], axis=0)
    torch.save(
        (img.astype(np.float32), np.squeeze(phi).astype(np.float32)), target_file
    )


def generate_dataset(dataset: str, n_samples: int, k: int, path: Optional[Path] = None):
    path = path or Path.cwd() / "datasets"

    path.mkdir(exist_ok=True)
    (path / "train").mkdir(exist_ok=True)
    (path / "test").mkdir(exist_ok=True)

    caesar_model = CaesarModel(Path(dataset))

    pl = pv.Plotter(window_size=(270, 480), off_screen=True)

    for i in tqdm(range(int(n_samples))):
        split = int(n_samples * 0.7)
        target_dataset = "train" if i < split else "test"
        target_file = path / target_dataset / f"sample{i % split}.pt"

        if not target_file.exists():
            if target_dataset == "train":
                with TemporaryDirectory() as tmpdir:
                    off_file = Path(tmpdir) / "tmp.off"
                    random_sample(pl, target_file, caesar_model, k, off_file)
            else:
                off_file = target_file.with_suffix(".off")
                random_sample(pl, target_file, caesar_model, k, off_file)

    pl.close()


class HumanImageDataset(Dataset):
    def __init__(self, img_dir, transform, target_transform):
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
        transform,
        target_transform,
        batch_size: int,
        shuffle: bool = True,
        split: float = 0.25,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["target_transform"])

        self.path = Path(path)
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
        generate_dataset("caesar-norm-wsx", 1000, 10, self.path)

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
        return DataLoader(
            self.test_dataset,
            self.batch_size
        )
