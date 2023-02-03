#!/bin/bash

# This bash script adds symbolic links to custom data locations.
# Update the external directories with your own data, if you have it.

set -e

DATAFOLDER=${1:-/data/$(whoami)}
DATAFOLDER=${DATAFOLDER%/}

THISDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

ln -sf "${DATAFOLDER}/KITTI" "$THISDIR/KITTI"
ln -sf "${DATAFOLDER}/nuScenes" "$THISDIR/nuScenes"
ln -sf "${DATAFOLDER}/nuImages" "$THISDIR/nuImages"
ln -sf "${DATAFOLDER}/CARLA" "$THISDIR/CARLA"