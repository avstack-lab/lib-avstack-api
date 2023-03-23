#!/usr/bin/env bash

set -e

# Training
open "https://drive.google.com/file/d/1DbBOURvIuV7E9_g4FpKwUGNJiIG_5Eeg/view?usp=share_link"

# Validation
open "https://drive.google.com/file/d/1M4pG-fdPs-EWMLZpc1yl-bqUcJ6yg4zz/view?usp=share_link"

# Testing
open "https://drive.google.com/file/d/1fuYK-oNA0FpZtT8rUiEETOCNmtO3FCfS/view?usp=share_link"


echo "Try as we may, we were not able to figure out a reliable way" \
     "to download large files from google drive through the comman" \
     "line (even with gdown). Therefore, instead, we have opened up" \
     "browser sessions at the links for the dataset folders. To use" \
     "these, you have to manually download them to any particular place" \
     "of your choosing. Then, you need to process the dataset by running" \
     "process_OPV2V_dataset.sh through the command line. You will need to" \
     "pass in the location where the dataset was downloaded."