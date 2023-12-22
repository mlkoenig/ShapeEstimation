"""
Module for the implementation of an inference class.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyvista as pv

from .models import CaesarModel, Encoder, ShapeModel


class InferenceModel:
    def __init__(self, caesar: CaesarModel, model: ShapeModel) -> None:
        """
        A class to predict on new front and side view images.

        Args:
            caesar (CaesarModel): The caesar model.
            model (ShapeModel): The trained shape model.
        """
        self.caesar = caesar
        self.model = model

    def __call__(self, x: Any) -> Any:
        """
        Predict the coefficent and vertex positions for a new input.

        Args:
            x (Any): The input image to the model.

        Returns:
            Tensor, Tensor: The predicted coeffs and the vertex positions.
        """
        y_pred = self.model(x)
        if self.model.inv_transform:
            y_pred = self.model.inv_transform(y_pred)

        return self.caesar(y_pred.cpu().detach().numpy())

    def predict(self, x) -> tuple[Any, Any]:
        """
        Predict the shape on an input image.

        Args:
            x (Any): The input image.

        Returns:
            Any, Any: The predicted coefficients and the pyvista mesh file.
        """
        y_pred, positions = self(x)

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / "tmp.off"
            self.caesar.to_off(tmp, positions)

            mesh = pv.read(tmp)

        return y_pred, mesh


def load_model(
    checkpoint: Path, caesar_path: Path, k: int = 10, sigma: float = 3.0, device=None
) -> InferenceModel:
    """
    Load a saved model from a checkpoint.

    Args:
        checkpoint (Path): The path to the checkpoint.
        caesar_path (Path): The path to the caesar dataset files.
        k (int, optional): The number of principal components. Defaults to 10.
        sigma (float, optional): The variance range. Defaults to 3.0.
        device (_type_, optional): The device to predict on. Defaults to None.

    Returns:
        InferenceModel: An inference model
    """
    caesar = CaesarModel(caesar_path, k=k, sigma=sigma)
    model = ShapeModel.load_from_checkpoint(
        str(checkpoint), device=device, model=Encoder()
    )
    return InferenceModel(caesar, model)
