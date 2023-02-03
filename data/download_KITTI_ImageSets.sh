#!/bin/bash

set -e

DATAFOLDER=${1:-/data/$(whoami)}
DATAFOLDER=${DATAFOLDER%/}
DATAFOLDER="${DATAFOLDER}/KITTI/object"

DOWNLOAD="https://avstack-public-data.s3.amazonaws.com/KITTI-ImageSets"

echo "Downloading KITTI ImageSets - saving to $DATAFOLDER"

files=(test.txt
train.txt
trainval.txt
val.txt)

mkdir -p "$DATAFOLDER/ImageSets"

for FILE in ${files[@]}; do
	echo "Downloading: $FILE"
        wget -P "$DATAFOLDER/ImageSets" "$DOWNLOAD/$FILE"
done