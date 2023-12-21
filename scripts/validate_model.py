from pathlib import Path

import pyvista as pv

from shapest.data import DataModule, target_transform
from shapest.inference import load_model
import matplotlib.pyplot as plt

ckpt = list(Path("lightning_logs/version_3").rglob("*.ckpt"))[0]

model = load_model(ckpt, caesar_path=Path("caesar-norm-wsx"))
device = model.model.device
datamodule = DataModule.load_from_checkpoint(ckpt, target_transform=target_transform)
datamodule.num_workers = 0
datamodule.batch_size = 1
datamodule.setup("test")


for i, (image, target) in enumerate(datamodule.test_dataloader()):
    image = image.to(device)
    target = target.to(device)

    y_pred, mesh2 = model.predict(image)

    mesh1 = pv.read(datamodule.path / "test" / f"sample{i}.off")

    pl = pv.Plotter(shape=(1, 3))
    pl.add_mesh(mesh1)
    pl.add_title("Ground Truth")

    pl.subplot(0, 1)
    pl.add_mesh(mesh2)
    pl.add_title("Predicted")

    pl.subplot(0, 2)
    pl.add_mesh(mesh1, opacity=0.5, color="green")
    pl.add_mesh(mesh2, opacity=0.5, color="red")
    pl.add_title("Difference")

    pl.link_views()

    pl.show()
