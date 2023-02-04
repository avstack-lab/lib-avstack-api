#!/usr/bin/env bash

set -e

VERSION=${1:-"object-v1"}  # object dataset --> "object-v1", collaborative dataset --> "collab-v1"
DATAFOLDER=${2:-/data/$(whoami)}
MAXFILES=${3:-10}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
DATAFOLDER="${DATAFOLDER}/CARLA"

DOWNLOAD="https://avstack-public-data.s3.amazonaws.com/CARLA-datasets"

if [ "$VERSION" = "object-v1" ]; then
    echo "Preparing to download object dataset..."
    SAVESUB="carla-object-v1"
    SUBDIR="object_v1"
    files=(run_2022_10_27_10:20:59
        run_2022_10_27_10:27:53
        run_2022_10_27_10:34:42
        run_2022_10_27_10:42:06
        run_2022_10_27_10:52:17
        run_2022_10_27_11:00:11
        run_2022_10_27_11:07:14
        run_2022_10_27_11:14:27
        run_2022_10_27_11:24:46
        run_2022_10_27_11:31:58
        run_2022_10_27_11:43:45
        run_2022_10_27_11:50:37
    )
elif [ "$VERSION" = "collab-v1" ]; then
    echo "Preparing to download collaborative dataset v1..."
    SAVESUB="carla-collaborative-v1"
    SUBDIR="object_collaborative_v1"
    files=(run_2022_10_31_13:34:52
        run_2022_10_31_13:46:21
    	run_2022_10_31_13:57:59
        run_2022_10_31_14:09:42
        run_2022_10_31_14:21:09
    )
elif [ "$VERSION" = "collab-v2" ]; then
    echo "Preparindg to download collaborative dataset v2..."
    SAVESUB="carla-collaborative-v2"
    SUBDIR="object_collaborative_v2"
    files=(run_2022_10_24_22:00:08
        run_2022_10_24_22:10:34
        run_2022_10_24_22:21:45
        run_2022_10_24_22:33:46
        run_2022_10_24_22:44:37
        run_2022_10_24_22:55:39
        run_2022_10_24_23:07:34
        run_2022_10_24_23:18:43
    )
else
    echo "Cannot understand input version ${VERSION}! Currently can only use 'object-v1' and 'collab-v1'"
fi

SAVEFULL="${DATAFOLDER}/${SAVESUB}"
mkdir -p $SAVEFULL

echo "Downloading up to $MAXFILES files"
COUNT=0

for FILE in ${files[@]}; do
    shortname="${FILE}.tar.gz"
    fullname="${SAVEFULL}/${shortname}"

    echo "Downloading ${shortname}"
    wget -P "$SAVEFULL" "$DOWNLOAD/$SAVESUB/$shortname"
    tar -xvf "$fullname" -C "$SAVEFULL" --force-local
    mv "$DATAFOLDER/$SAVESUB/$SUBDIR/$FILE" "$DATAFOLDER/$SAVESUB/$FILE"  # this is a result of a saving error previously
    rm -r "$DATAFOLDER/$SAVESUB/$SUBDIR"
    COUNT=$((COUNT+1))
    echo "Downloaded $COUNT / $MAXFILES files!"
    if [[ $COUNT -ge $MAXFILES ]]; then
            echo "Finished downloading $COUNT files"
            break
    fi
done
