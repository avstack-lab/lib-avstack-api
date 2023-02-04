#!/bin/bash

set -e

DATAFOLDER=${1:-/data/$(whoami)}
MAXFILES=${2:-38}

DATAFOLDER=${DATAFOLDER%/}
DATAFOLDER="${DATAFOLDER}/KITTI/raw"

files=(2011_09_26_drive_0001
2011_09_26_drive_0002
2011_09_26_drive_0005
2011_09_26_drive_0009
2011_09_26_drive_0011
2011_09_26_drive_0013
2011_09_26_drive_0014
2011_09_26_drive_0015
2011_09_26_drive_0017
2011_09_26_drive_0018
2011_09_26_drive_0019
2011_09_26_drive_0020
2011_09_26_drive_0022
2011_09_26_drive_0023
2011_09_26_drive_0027
2011_09_26_drive_0028
2011_09_26_drive_0029
2011_09_26_drive_0032
2011_09_26_drive_0035
2011_09_26_drive_0036
2011_09_26_drive_0039
2011_09_26_drive_0046
2011_09_26_drive_0048
2011_09_26_drive_0051
2011_09_26_drive_0052
2011_09_26_drive_0056
2011_09_26_drive_0057
2011_09_26_drive_0059
2011_09_26_drive_0060
2011_09_26_drive_0061
2011_09_26_drive_0064
2011_09_26_drive_0070
2011_09_26_drive_0079
2011_09_26_drive_0084
2011_09_26_drive_0086
2011_09_26_drive_0087
2011_09_26_drive_0091
2011_09_26_drive_0093)

mkdir -p "$DATAFOLDER"

echo "Downloading up to $MAXFILES files"
COUNT=0

for FILE in ${files[@]}; do
        if [ ${FILE:(-3)} != "zip" ]
        then
                shortname="${FILE}_tracklets.zip"
                fullname="${FILE}/${FILE}_tracklets.zip"
        else
                continue
        fi
        echo "Downloading: "$shortname
        wget -P "$DATAFOLDER" "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/${fullname}"
        if [ $? -eq 0 ]; then
                unzip -o "$DATAFOLDER/$shortname" -d "$DATAFOLDER"
                rm "${DATAFOLDER}/${shortname}"
        else
                echo "This download failed!!!"
        fi
        COUNT=$((COUNT+1))
        echo "Downloaded $COUNT / $MAXFILES files!"
        if [[ $COUNT -ge $MAXFILES ]]; then
                echo "Finished downloading $COUNT files"
                break
        fi
done