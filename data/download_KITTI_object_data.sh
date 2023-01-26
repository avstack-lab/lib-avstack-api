#!/bin/bash

set -e

SAVEFOLDER="${1:-"./KITTI/object"}"
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash

LABEL_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
CALIB_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
LIDAR_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
IMAG2_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"

echo "Downloading KITTI object dataset - saving to $SAVEFOLDER"

DATA_PATHS=($LABEL_PATH
            $CALIB_PATH
            $LIDAR_PATH
            $IMAG2_PATH)

mkdir -p "$SAVEFOLDER"

for DATA_PATH in ${DATA_PATHS[@]}
do
    wget -P "$SAVEFOLDER" "$DATA_PATH"
    if [ $? -eq 0 ]; then
        shortname="${DATA_PATH##*/}"
        unzip -o "$SAVEFOLDER/$shortname" -d "$SAVEFOLDER"
        rm "${SAVEFOLDER}/${shortname}"
    else
        echo "This download failed!!!"
    fi
done
cd ../..
