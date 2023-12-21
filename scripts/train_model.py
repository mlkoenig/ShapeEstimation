from pathlib import Path

from lightning.pytorch import Trainer

from shapest.data import DataModule, inv_transform, target_transform
from shapest.models import Encoder, ShapeModel

if __name__ == "__main__":
    data_path = Path(__file__).parents[1].resolve() / "datasets"
    datamodule = DataModule(
        data_path,
        transform=None,
        target_transform=target_transform,
        batch_size=128,
        num_workers=4,
    )

    model = ShapeModel(Encoder(), inv_transform)
    trainer = Trainer(max_epochs=2, enable_checkpointing=True, precision="32")

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
