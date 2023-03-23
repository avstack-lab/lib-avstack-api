#!/usr/bin/env bash

set -e

DOWNLOADFOLDER=${1:-"/home/$(whoami)/Downloads"}
DATAFOLDER=${2:-/data/$(whoami)}

DOWNLOADFOLDER=${DOWNLOADFOLDER%/}  # remove trailing slash
DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash

DATAFOLDER="${DATAFOLDER}/OPV2V"
mkdir -p "$DATAFOLDER"

SAVEFULL="${DATAFOLDER}"
files=(train
       validate 
       test)

for FILE in ${files[@]}; do
    zipfile="${DOWNLOADFOLDER}/${FILE}"
    echo "Unzipping $zipfile"
    unzip $zipfile -d "${SAVEFULL}"
done
