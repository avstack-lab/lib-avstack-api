#!/bin/bash

set -e

DATAFOLDER=${1:-/data/$(whoami)}
DATAFOLDER=${DATAFOLDER%/}
DATAFOLDER="${DATAFOLDER}/nuScenes"

evidence="${DATAFOLDER}/can_bus"

if [ -d "$evidence" ]; then
        echo -e "\nAlready downloaded nuScenes CAN bus data\n"
else
    mkdir -p "$DATAFOLDER"
    wget -P "$DATAFOLDER" "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/can_bus.zip"

    echo "Unzipping CAN bus files..."
    unzip "$DATAFOLDER/can_bus.zip" -d "$DATAFOLDER"
    rm "$DATAFOLDER/can_bus.zip"
    echo "done!"
fi
