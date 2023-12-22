"""
Train a CNN to estimate the coefficients of the PCA eigen vectors.
"""

from pathlib import Path

from lightning.pytorch import Trainer

from shapest import DataModule, ShapeModel
from shapest.models import Encoder

if __name__ == "__main__":
    root = Path(__file__).parents[1].resolve()
    datamodule = DataModule(
        root / "datasets",
        root / "caesar-norm-wsx",
        batch_size=128,
        num_workers=4,
    )

    model = ShapeModel(Encoder())
    trainer = Trainer(max_epochs=20, enable_checkpointing=True, precision="32")

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
