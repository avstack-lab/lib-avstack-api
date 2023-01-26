#!/bin/bash

set -e

SAVEFOLDER="${1:-"./nuScenes"}"
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash

mkdir -p "$SAVEFOLDER"
wget -P "$SAVEFOLDER" "https://www.nuscenes.org/data/v1.0-mini.tgz"

echo "Extracting files..."
tar -xf "$SAVEFOLDER/v1.0-mini.tgz" -C "$SAVEFOLDER" --remove-files
rm "$SAVEFOLDER/v1.0-mini.tgz"
echo "done!"