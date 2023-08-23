#!/bin/bash

set -e

SAVEFOLDER=${1:-/data/tracking}
SAVEFOLDER=${SAVEFOLDER%/}  # remove trailing slash

DATALINK="https://motchallenge.net/data/MOT15.zip"
DATAFILE="MOT15.zip"
DATASUBFOLDER="MOT15"

mkdir -p $SAVEFOLDER
wget -P $SAVEFOLDER $DATALINK
unzip -o "${SAVEFOLDER}/${DATAFILE}" -d "$SAVEFOLDER"
rm "${SAVEFOLDER}/${DATAFILE}"

ln -s "${SAVEFOLDER}/${DATASUBFOLDER}" "./MOT15"
