#!/usr/bin/env bash

set -xe

# python postprocess_carla_objects.py ../../examples/lidar-results

# mv ../../examples/lidar-results/run* /data/spencer/CARLA/ego-lidar/

cd ../avstack-core/third_party/mmdetection3d/CUSTOM

./convert_carla_vehicle.sh

cd ..

./CUSTOM/run_train_carla_vehicle.sh pointpillars
