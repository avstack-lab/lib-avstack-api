#!/bin/bash

set -e

mkdir -p nuScenes
cd nuScenes
wget https://www.nuscenes.org/data/v1.0-mini.tgz
tar -xf v1.0-mini.tgz
rm v1.0-mini.tgz
cd ..