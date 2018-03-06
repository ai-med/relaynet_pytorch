#!/usr/bin/env bash

REL_SCRIPT_DIR=$(dirname "$0")
INITIAL_DIR=$(pwd)
cd $REL_SCRIPT_DIR
ABS_SCRIPT_DIR=$(pwd)

cd datasets
rm -r segmentation_data segmentation_data_test cifar10_train.p

# Get CIFAR10
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/cifar10_train.zip
tar -xzvf cifar10_train.zip
rm cifar10_train.zip

# Get segmentation dataset
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data.zip
unzip segmentation_data.zip
rm segmentation_data.zip

# Get segmentation dataset test
wget http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data_test.zip
unzip segmentation_data_test.zip
rm segmentation_data_test.zip

cd $INITIAL_DIR