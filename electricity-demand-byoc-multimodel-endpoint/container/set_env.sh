#!/bin/bash -ex

WORKING_DIR=./.myenv
ENV_NAME=gluonts-multimodel
mkdir -p "${WORKING_DIR}"

# Install Miniconda to get a separate python and pip
wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O "$WORKING_DIR/miniconda.sh"

# Install Miniconda into the working directory
bash "$WORKING_DIR/miniconda.sh" -b -u -p "$WORKING_DIR/miniconda"

# Install pinned versions of any dependencies
source "$WORKING_DIR/miniconda/bin/activate"

conda create -y -n $ENV_NAME anaconda python=3.6

conda activate $ENV_NAME

## Install required library
pip install -r requirements.txt

## install Prophet python packages
conda install -c conda-forge fbprophet --yes

## install gluonts
pip install gluonts==0.6.7

# Cleanup
conda deactivate
source "${WORKING_DIR}/miniconda/bin/deactivate"
rm -rf "${WORKING_DIR}/miniconda.sh"