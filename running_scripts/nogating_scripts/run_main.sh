#!/bin/bash

PORT=$1
METHOD=$2
START_GPU=$3
NUM_GPU=$4
BACKBONE=$5

# ./run_quad.sh 6000 nogate 7 8 resnet50
./run_quad.sh 6000 static 7 8 resnet50
