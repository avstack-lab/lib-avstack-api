#!/bin/bash

set -e

DATAFOLDER=${1:-/data/$(whoami)}
DATAFOLDER=${DATAFOLDER%/}
DATAFOLDER="${DATAFOLDER}/KITTI/object"

LABEL_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
CALIB_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
LIDAR_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
IMAG2_PATH="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"

echo "Downloading KITTI object dataset - saving to $DATAFOLDER"

DATA_PATHS=($LABEL_PATH
            $CALIB_PATH
            $LIDAR_PATH
            $IMAG2_PATH)

mkdir -p "$DATAFOLDER"

for DATA_PATH in ${DATA_PATHS[@]}
do
    shortname="${DATA_PATH##*/}"
    fol_name="${shortname//data_object_/}"
    fol_name="${fol_name//.zip/}"
    fol_name="${fol_name%/}"
    evidence="${DATAFOLDER}/training/${fol_name}/.full_download"
    if [ -f "$evidence" ]; then
        echo -e "$fol_name already downloaded.\n"
    else
        wget -P "$DATAFOLDER" "$DATA_PATH"
        unzip -o -d "$DATAFOLDER" "$DATAFOLDER/$shortname"
        rm "${DATAFOLDER}/${shortname}"
        touch $evidence
    fi
done