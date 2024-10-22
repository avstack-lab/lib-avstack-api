#!/usr/bin/env bash

set -e

VERSION=${1:-"multi-agent-v1"}
DATAFOLDER=${2:-/data/$(whoami)}
MAXFILES=${3:-10}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
DATAFOLDER="${DATAFOLDER}/CARLA"

DOWNLOAD="https://g-b0ef78.1d0d8d.03c0.data.globus.org/datasets/carla"

if [ "$VERSION" = "multi-agent-v1" ]; then
    echo "Preparing to download ego-lidar dataset..."
    SAVESUB="multi-agent-v1"
    SUBDIR="multi-agent-v1"
    files=(
        run_2022_10_27_10-20-59
        run_2022_10_27_10-27-53
        run_2022_10_27_10-34-42
        run_2022_10_27_10-42-06
        run_2022_10_27_10-52-17
        run_2022_10_27_11-00-11
        run_2022_10_27_11-07-14
        run_2022_10_27_11-14-27
        run_2022_10_27_11-24-46
        run_2022_10_27_11-31-58
        run_2022_10_27_11-43-45
        run_2022_10_27_11-50-37
    )
else
    echo "Cannot understand input version ${VERSION}! Currently can only use 'multi-agent-v1'"
fi

SAVEFULL="${DATAFOLDER}/${SAVESUB}"
mkdir -p $SAVEFULL

echo "Downloading up to $MAXFILES files"
COUNT=0

for FILE in ${files[@]}; do
    shortname="${FILE}.tar.gz"
    fullname="${SAVEFULL}/${shortname}"
    F_REP="${FILE//-/:}"
    evidence="${SAVEFULL}/${F_REP}/.full_download"

    # -- check for evidence of full download
    if [ -f "$evidence" ]; then
        echo -e "$shortname exists and already unzipped\n"
    # -- check for existing tar.gz file
    elif [ -f "$DOWNLOAD/$SAVESUB/$shortname" ]; then
        echo -e "$shortname exists...unzipping\n"
        tar -xvf "$fullname" -C "$SAVEFULL" --force-local
        mv "$DATAFOLDER/$SAVESUB/$SUBDIR/$FILE" "$DATAFOLDER/$SAVESUB/$FILE"  # this is a result of a saving error previously
        rm -r "$DATAFOLDER/$SAVESUB/$SUBDIR"
        rm "$DATAFOLDER/$SAVESUB/${FILE}.tar.gz"
    # -- otherwise, download again
    else
        echo "Downloading ${shortname}"
        wget -P "$SAVEFULL" "$DOWNLOAD/$SAVESUB/$shortname"
        tar -xvf "$fullname" -C "$SAVEFULL" --force-local
        mv "$DATAFOLDER/$SAVESUB/$SUBDIR/$F_REP" "$DATAFOLDER/$SAVESUB/$F_REP"  # this is a result of a saving error previously
        rm -r "$DATAFOLDER/$SAVESUB/$SUBDIR"
        rm "$DATAFOLDER/$SAVESUB/${FILE}.tar.gz"
    fi
    
    # -- add evidence of successful download
    touch "$evidence"

    # -- check downloads
    COUNT=$((COUNT+1))
    echo "Downloaded $COUNT / $MAXFILES files!"
    if [[ $COUNT -ge $MAXFILES ]]; then
            echo "Finished downloading $COUNT files"
            break
    fi
done
