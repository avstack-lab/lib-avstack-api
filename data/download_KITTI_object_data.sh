#!/bin/bash

LABEL_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
CALIB_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
LIDAR_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
IMAG2_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"

DATA_PATHS=($LABEL_PATH
            $CALIB_PATH
            $LIDAR_PATH
            $IMAG2_PATH)

mkdir -p KITTI/object
cd KITTI/object

for DATA_PATH in ${DATA_PATHS[@]}
do
    wget "$DATA_PATH"
    if [ $? -eq 0 ]; then
        shortname="${DATA_PATH##*/}"
        unzip -o $shortname
        rm $shortname
    else
        echo "This download failed!!!"
    fi
done
cd ../..
