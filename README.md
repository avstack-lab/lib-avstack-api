# AVstack API

This is the API library of `AVstack`.  It relies on both [`avstack-core`][avstack-core] and having access to integrated datasets/simulators. `AVstack` was published and presented at ICCPS 2023 - [find the paper here][avstack-preprint] that accompanies the repository.

## Philosophy

The `AVstack` API unlocks consistent interfacing to some of the most popular and useful autonomous vehicle driving datasets. The goal of the `AVstack` API is to standardize interfacing between data source providers. This enables better transfer between datasets and/or simulators and helps unify disparate AV conventions.

See [`avstack-core`][avstack-core]

## Installation

First, clone the repositry and submodules.
```
git clone --recurse-submodules https://github.com/avstack-lab/avstack-api.git 
```
Dependencies are managed with [`poetry`][poetry]. This uses the `pyproject.toml` file to create a `poetry.lock` file. 

You must already have downloaded [`avstack-core`][avstack-core] and placed it in a location commensurate with this repository's `pyproject.toml` file. For example, if `pyproject.toml` says 
```
avstack-core = { path = "../avstack-core", develop = true }
```
then `avstack-core` must be placed in the same folder as `avstack-api`. 

### Datasets

See [the dataset README file](https://github.com/avstack-lab/avstack-api/data/README.md)

### Simulators

`AVstack` is also compatible with autonomous vehicle simulators.

#### CARLA

See the [carla-sandbox][carla-sandbox] to get started with a development sandbox.

## Demos

We have provided a few Jupyter notebooks in the `notebooks/` folder. Please refrain from committing large jupyter notebooks in pull requests. Either ignore them or use something like [this approach](https://zhauniarovich.com/post/2020/2020-10-clearing-jupyter-output-p2/).

## Usage Notes

### Git Hooks

To prevent the inflation of the repository due to e.g., image data in the example notebooks, please configure a pre-commit hook to clear the output of the jupyter notebooks. To make this process easier, we've included a `hooks` directory. By running:
```
git config core.hooksPath hooks
```
in the project root, you can use our pre-made pre-commit hook.

# Contributing

See [CONTRIBUTING.md](https://github.com/avstack-lab/avstack-core/CONTRIBUTING.md) for further details.

# LICENSE

Copyright 2023 Spencer Hallyburton

AVstack specific code is distributed under the MIT License.


[poetry]: https://github.com/python-poetry/poetry
[avstack-core]: https://github.com/avstack-lab/avstack-core
[avstack-preprint]: https://arxiv.org/pdf/2212.13857.pdf
[carla-sandbox]: https://github.com/avstack-lab/carla-sandbox
