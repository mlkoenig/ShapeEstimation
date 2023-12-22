"""
Visualize and compare the model shape estimations with the test samples.
"""

from pathlib import Path

import click
import numpy as np
import pyvista as pv

from shapest import DataModule, load_model


@click.command()
@click.argument(
    "ckpt_path", type=click.Path(exists=True, file_okay=True, dir_okay=False)
)
def main(ckpt_path: str):
    datamodule = DataModule.load_from_checkpoint(ckpt_path)
    datamodule.num_workers = 0
    datamodule.batch_size = 1
    datamodule.setup("test")

    model = load_model(Path(ckpt_path), caesar_path=datamodule.caesar, k=10, sigma=3.0)
    device = model.model.device

    for i, (image, target) in enumerate(datamodule.test_dataloader()):
        image = image.to(device)
        target = target.to(device)

        _, mesh1 = model.predict(image)

        mesh2 = pv.read(datamodule.path / "test" / f"sample{i}.off")
        mesh2["Distance error"] = np.linalg.norm(mesh1.points - mesh2.points, axis=1)

        pl = pv.Plotter(shape=(1, 3))
        pl.add_mesh(mesh1)
        pl.add_title("Predicted")

        pl.subplot(0, 1)
        pl.add_mesh(mesh2, cmap="coolwarm")
        pl.add_title("Ground Truth")

        pl.subplot(0, 2)
        pl.add_mesh(mesh1, opacity=0.5, color="red")
        pl.add_mesh(mesh2, opacity=0.5, color="green")
        pl.add_title("Difference")

        pl.link_views()

        pl.show()


if __name__ == "__main__":
    main()
