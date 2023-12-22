# Human Shape Estimation

This repository aims to estimate a fitting 3D mesh of a human from front and side view silhouettes using the [MPII Human Shape](https://humanshape.mpi-inf.mpg.de) models from the paper "[Building Statistical Shape Spaces for 3D Human Modeling](https://arxiv.org/abs/1503.05860)" by `Leonid Pishchulin et al. (2015)`, who provide standardized meshes with fitted vertex positions per scan, the so-called shape spaces.

The main idea is to estimate the first `k` eigenvalues of the Principal Component Analysis (PCA) given by the dataset.
This is done by processing the front and side view images with a Convolutional Neural Network (CNN) implemented with PyTorch Lightning.

The estimated eigenvalues are used to reconstruct the correct vertex positions of the standardized mesh.

## Usage

### Installation

After cloning the repository, the package can be installed in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Creating training and test data

A `CAESAR` dataset needs to be provided in order to create the training and test data.

:warning: **By downloading and using the human shape spaces, you agree to the following license** (see [bsd.txt](https://humanshape.mpi-inf.mpg.de/bsd.txt)):

```
MPII Human Shape, Version 1.0
Copyright 2015 Max Planck Institute for Informatics
Licensed under the Simplified BSD License

The shape spaces and fitted scans are freely available for non-commercial and educational purposes only as of 22.12.2023. The authors need to be asked for authorization for any other purposes.
```

Use one of the datasets as described in the [HUMANSHAPE repository](https://github.com/leonid-pishchulin/humanshape/tree/master), create the corresponding `evalues.mat`, `evectors.mat`, and `meanShape.mat` files and put them into a folder, e.g. `<project root>/caesar-norm-wsx`.
Afterwards, copy the `shapemodel/model.dat` from the repository into the same folder.

Now, run the data generation to create a training and test dataset.
```bash
python scripts/generate_datasets.py <caesar dataset path> -n 2000 -k 10
```
Validate the generation by observing the newly created `datasets` folder, which should have `train` and `test` subdirectories with a 30% split of the original sampling size.

### Training a model

Now that the training and test data is generated we can start the training.

Run the training script:

```bash
python scripts/train_model.py
```

Checkpointing is enabled by default. When the training is finished, the checkpoints can be located at `lightning_logs/version_x/checkpoints`.

### Testing and visualizing the shape estimations

We use [pyvista](https://pyvista.org) to process and render the mesh data.

Run the test script to see a side-by-side comparison of
the ground truth test samples and the model estimations:

```bash
python scripts/test_model.py <checkpoint_file> <caesar_dataset>
```
<img width="505" alt="Bildschirmfoto 2023-12-22 um 15 17 06" src="https://github.com/mkoenig-dev/ShapeEstimation/assets/51786860/8b6469f3-5456-4187-a0c2-2072c75342ff">
