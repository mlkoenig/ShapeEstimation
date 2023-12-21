from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyvista as pv

from .data import transform, inv_transform
from .models import CaesarModel, ShapeModel, Encoder


class InferenceModel:
    def __init__(self, caesar: CaesarModel, model: ShapeModel) -> None:
        self.caesar = caesar
        self.model = model

    def __call__(self, x) -> Any:
        y_pred = self.model(x)
        if self.model.inv_transform:
            y_pred = self.model.inv_transform(y_pred)

        return y_pred, self.caesar.predict(y_pred.cpu().detach().numpy())

    def predict(self, x) -> pv.PolyData:
        y_pred, positions = self(x)

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "tmp.off"
            self.caesar.to_off(tmp, positions)

            mesh = pv.read(tmp)

        return y_pred, mesh


def load_model(checkpoint: Path, caesar_path: Path, device=None) -> InferenceModel:
    caesar = CaesarModel(caesar_path)
    model = ShapeModel.load_from_checkpoint(str(checkpoint), device=device, model=Encoder(), inv_transform=inv_transform)
    return InferenceModel(caesar, model)
