#!/usr/bin/env bash

CONFIG=configs/softgroup/softgroup_scannet.yaml
CKPT=ckpt/softgroup_scannet_spconv2.pth
GPUS=1
OUT_DIR=results/

./tools/dist_test.sh $CONFIG $CKPT $GPUS --out $OUT_DIR