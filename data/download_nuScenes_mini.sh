#!/bin/bash

set -e

DATAFOLDER=${1:-/data/$(whoami)}
DATAFOLDER=${DATAFOLDER%/}
DATAFOLDER="${DATAFOLDER}/nuScenes"

mkdir -p "$DATAFOLDER"
wget -P "$DATAFOLDER" "https://www.nuscenes.org/data/v1.0-mini.tgz"

echo "Extracting files..."
tar -xf "$DATAFOLDER/v1.0-mini.tgz" -C "$DATAFOLDER" --remove-files
rm "$DATAFOLDER/v1.0-mini.tgz"
echo "done!"