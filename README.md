# SfPlanar

This repository provides the source code for the paper **Variational Inference with Singularity-Free Planar Flows**, accepted for publication in *Machine Learning*.

SfPlanar is a singularity-free variant of planar normalizing flows, developed to address numerical instability in conventional planar flows.

---

## Repository Structure

```text
experiments/     Code to reproduce the experiments in the paper
src/sfplanar/    Package source code
pyproject.toml   Package configuration and dependencies
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/jr-ouyang/SfPlanar.git
cd SfPlanar
```

Create and activate a Python environment:

```bash
mamba create -n sfplanar python=3.13
mamba activate sfplanar
```

Install the package:

```bash
python -m pip install .
```

All Python dependencies are declared in [pyproject.toml](pyproject.toml) and will be installed automatically.

---

## Experiments

The [`experiments/`](experiments) directory contains code to reproduce the numerical experiments reported in the paper.

The workflow is managed using [Snakemake 9.20.0](https://snakemake.github.io/) together with the [Snakemake executor plugin for Slurm 2.6.1](https://github.com/snakemake/snakemake-executor-plugin-slurm). 

The rules and settings for Slurm job submission are defined in each subfolder's `Snakefile`.


