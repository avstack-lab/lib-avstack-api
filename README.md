# AVstack API

This is the API library of `AVstack`.  It relies on both [`avstack-core`][avstack-core] and having access to integrated datasets/simulators. This accompanies the preprint [HERE][avstack-preprint].

## Philosophy

The `AVstack` API unlocks consistent interfacing to some of the most popular and useful autonomous vehicle driving datasets. The goal of the `AVstack` API is to standardize interfacing between data source providers. This enables better transfer between datasets and/or simulators and helps unify disparate AV conventions.

See [`avstack-core`][avstack-core]

## Installation

First, clone the repositry and submodules.
```
git clone --recurse-submodules https://github.com/avstack-lab/lib-avstack-api.git 
```
Dependencies are managed with [`poetry`][poetry]. This uses the `pyproject.toml` file to create a `poetry.lock` file. 

You must already have downloaded [`avstack-core`][avstack-core] and placed it in a location commensurate with this repository's `pyproject.toml` file. For example, if `pyproject.toml` says 
```
lib-avstack-core = { path = "../lib-avstack-core", develop = true }
```
then `lib-avstack-core` must be placed in the same folder as `lib-avstack-api`. 

### Datasets

To make use of the power of the `AVstack` API, you must download and organize some autonomous vehicle datasets. Around the internet, you will find many a number of ways AV developers have chosen to organize their AV datasets. It's important to follow the convention outlined here (or suggest a better one). 

- **Option 1:** Use the full path to an existing download of the data.
- **Option 2:** Use the utilities in the `data` folder to download a copy of the data. This is described below.

*all code is relative to data folder*

#### KITTI

##### Object Dataset
*this is optional - you can do your own download elsewhere*
```
./download_KITTI_object_data.sh
```

##### Raw Data
*this is optional - you can do your own download elsewhere*
```
./download_KITTI_raw_tracklets.sh
./download_KITTI_raw_data.sh
```
Some of the tracklet downloads may fail. This is not a problem.

#### nuScenes mini
*this is optional - you can do your own download elsewhere*
```
./download_nuScenes_mini.sh
```

#### nuScenes, nuImages, Waymo
These all require an AWS access key obtained through your login to their website. Just follow the standard download procedure and place somewhere you can identify. 

### Simulators

`AVstack` is also compatible with autonomous vehicle simulators.

#### CARLA

See the [carla-sandbox][carla-sandbox] to get started with a development sandbox.

## Demos

We have provided a few Jupyter notebooks in the `notebooks/` folder. Please refrain from committing large jupyter notebooks in pull requests. Either ignore them or use something like [this approach](https://zhauniarovich.com/post/2020/2020-10-clearing-jupyter-output-p2/).

# Contributing

See [CONTRIBUTING.md](https://github.com/avstack-lab/lib-avstack-core/CONTRIBUTING.md) for further details.

# LICENSE

Copyright 2022 Spencer Hallyburton

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Logo by [Sydney Jeffs](https://twitter.com/sydney_jeffs).




[avstack-core]: https://github.com/avstack-lab/lib-avstack-core
[avstack-preprint]: todo
[carla-sandbox]: https://github.com/avstack-lab/carla-sandbox
