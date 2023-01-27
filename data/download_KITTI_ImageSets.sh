#!/bin/bash

set -e

SAVEFOLDER="${1:-"./KITTI/object"}"
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash

DATAFOLDER="https://avstack-public-data.s3.amazonaws.com/KITTI-ImageSets"

echo "Downloading KITTI ImageSets - saving to $SAVEFOLDER"

files=(test.txt
train.txt
trainval.txt
val.txt)

mkdir -p "$SAVEFOLDER/ImageSets"

for FILE in ${files[@]}; do
	echo "Downloading: $FILE"
        wget -P "$SAVEFOLDER/ImageSets" "$DATAFOLDER/$FILE"
done