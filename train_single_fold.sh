#!/bin/bash

set -e

if [ -z "$1" ]
    then
    echo "No argument supplied"
    exit 1
fi

exp_name=cross_val_test
dataset=cross_val
iFold=$1
train_set_name=scripts/split/${dataset}/${iFold}_train.csv
val_set_name=scripts/split/${dataset}/${iFold}_val.csv
test_set_name=$val_set_name
mask_out_dir=results/${exp_name}/Fold_${iFold}_mask
#rcnn_out_dir=results/${exp_name}/${iFold}_rcnn
#rpn_out_dir=results/${exp_name}/${iFold}_rpn
mask_ckpt_path=${mask_out_dir}/model/200.ckpt
#rcnn_ckpt_path=${rcnn_out_dir}/model/200.ckpt
#rpn_ckpt_path=${rpn_out_dir}/model/200.ckpt

# Training with mask

if [ -z "$2" ]
  then
    python train.py --train-set-list $train_set_name --val-set-list $val_set_name --out-dir $mask_out_dir #--epoch-rcnn 65 --epoch-mask 80
  else
    checkpoint=$2
    python train.py --train-set-list $train_set_name --val-set-list $val_set_name --out-dir $mask_out_dir --ckpt $2 #--epoch-rcnn 65 --epoch-mask 80
fi
