#!/bin/bash

# This bash script adds symbolic links to custom data locations.
# Update the external directories with your own data, if you have it.

set -e

ln -sf /data/spencer/KITTI KITTI
ln -sf /data/spencer/nuScenes nuScenes
ln -sf /data/spencer/nuImages nuImages
ln -sf /data/spencer/CARLA CARLA