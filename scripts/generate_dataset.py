"""
Create a train/test dataset, where 30% of the total samples are
corresponding to the test dataset.
"""

from pathlib import Path
from typing import Optional

import click

from shapest import generate_dataset


@click.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=True, file_okay=False))
@click.option("--n_samples", "-n", type=int, default=10)
@click.option("-k", type=int, default=10)
@click.option("--sigma", type=float, default=3.0)
def main(
    dataset: str, n_samples: int, k: int, sigma: float, path: Optional[Path] = None
):
    generate_dataset(dataset, n_samples, k=k, sigma=sigma, path=path)


if __name__ == "__main__":
    main()
