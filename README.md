# Sequential Kernel-based Conditional Independence Testing via Adaptive Betting

This repository contains the code for the ICML 2026 paper on ``Sequential Kernel-based Conditional Independence Testing via Adaptive Betting''.

## Overview

Conditional independence testing is a fundamental problem in statistics and machine learning, but without additional assumptions, valid Type I error control is impossible in full generality. The Model-X framework addresses this by assuming access to the relevant conditional distribution, yet many existing sequential conditional independence tests remain fragile when that conditional must be estimated rather than known exactly.

This repository implements a more robust sequential testing approach based on:

- testing-by-betting,
- an adaptively optimized kernel conditional independence statistic,
- a normalization scheme, and
- a truncate-and-shift calibration strategy.

Across synthetic benchmarks and real-data fairness tasks, the method is designed to substantially reduce Type I error inflation while maintaining high power under alternatives.

## Repository Structure

- `train.py`: main training and experiment entry point.
- `configs/experiment/`: experiment presets.
- `configs/data/`: dataset configuration files.
- `configs/kernel/`: kernel and feature model configuration files.
- `trainer/`: training loop and betting procedure.
- `data/`: synthetic and real-data generators.
- `models/`: kernel models, CNN/MLP modules, and autoencoder components.
- `utils/`: kernel computations, losses, and matrix processing utilities.

## Requirements

The repository was developed with a pinned environment in `requirements.txt`, but exact version matching is not necessary in most cases. A setup roughly satisfying the following should be sufficient:

- Python `>= 3.10`
- PyTorch `>= 2.0`
- torchvision `>= 0.15`
- numpy `>= 1.24`
- scipy `>= 1.10`
- pandas `>= 2.0`
- hydra-core `>= 1.3`
- omegaconf `>= 2.3`
- wandb `>= 0.15`

If you want to reproduce the original environment more closely, install directly from `requirements.txt`.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Experiments

Experiments are configured with Hydra. The main entry point is:

```bash
python train.py +experiment=<name>
```

### Synthetic Benchmarks

Gaussian benchmark:

```bash
python train.py +experiment=gaussian
```

Conditional independence hardness / sinusoidal benchmark:

```bash
python train.py +experiment=ci_hardness
```

Useful overrides:

```bash
python train.py +experiment=gaussian train.seed=2 data.samples=10 train.seqs=200
python train.py +experiment=ci_hardness data.d=3 data.ca_dim_idx=0 data.cb_dim_idx=1 data.cr_dim_idx=2
```

### Real and Semi-Real Data Experiments

RatInABox benchmark:

```bash
python train.py +experiment=ratinabox data.data_path=/path/to/ratinabox/data
```

dSprites benchmark:

```bash
python train.py +experiment=dsprites data.data_path=/path/to/dsprites_ndarray.npz
```

Car insurance fairness benchmark:

```bash
python train.py +experiment=carinsurance data.data_path=/path/to/car_insurance
```

Example overrides for car insurance:

```bash
python train.py +experiment=carinsurance data.data_path=/path/to/car_insurance data.state=ca data.company_idx=1
python train.py -m +experiment=carinsurance data.data_path=/path/to/car_insurance data.state=ca,il,mo,tx
```

## Datasets

The repository includes both self-contained synthetic generators and experiments that rely on external datasets.

- `gaussian`: fully synthetic, no external files required.
- `ci_hardness`: fully synthetic, no external files required.
- `ratinabox`: requires pre-generated `.npy` simulation files.
- `dsprites`: requires the dSprites `.npz` dataset.
- `carinsurance`: requires the car insurance CSV data.

Some data config files currently point to local machine paths under `data/...`. In practice, you should override these paths from the command line rather than editing the source configs.


The following datasets and experiment setups are adapted from prior repositories:

- `ratinabox` and the related experiments are adopted from [romanpogodin/kernel-ci-testing](https://github.com/romanpogodin/kernel-ci-testing/tree/main).
- `carinsurance` follows [felipemaiapolo/cit](https://github.com/felipemaiapolo/cit.git) and re-uses some of their code, as noted in the corresponding source files.
- `dsprites` uses the [google-deepmind/dsprites-dataset](https://github.com/google-deepmind/dsprites-dataset).


## Configuration Options

Some of the most useful Hydra overrides are:

- `train.seed=<int>`: random seed for training and experiment generation.
- `data.samples=<int>`: number of samples per sequence step.
- `train.seqs=<int>`: number of sequential batches.
- `train.T=<int>`: warm-start batches used for training only.
- `train.model_x_mode=online|pretrained|oracle`: how the regression components are fit.
- `wandb.disabled=true`: disable Weights & Biases logging.

For example:

```bash
python train.py +experiment=gaussian wandb.disabled=true train.model_x_mode=online
```

By default, experiment presets enable Weights & Biases logging. If you do not want external logging, disable it explicitly:

```bash
python train.py +experiment=gaussian wandb.disabled=true
```

## Citation

If you find this repository helpful, please consider cite
```
@inproceedings{he2026skci,
  title={Sequential Kernel-based Conditional Independence Testing via Adaptive Betting},
  author={He, Zheng and Sutherland, Danica J},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```
