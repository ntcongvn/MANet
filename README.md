# MANet
MANet : Multi-branch attention auxiliary learning for lung nodule detection and segmentation

##  Introduction

This repository contains the PyTorch implementation of MANet, Multi-branch attention auxiliary learning for lung nodule detection and segmentation. 

##  Install dependencies

Dependent libraries
* torch
* torchvision 
* opencv
* tqdm
* tb-nightly
* pynrrd
* SimpleITK
* pydicom

Install a custom module for bounding box NMS and overlap calculation.

```bask
cd build/box
python setup.py install
```


##  Usage

####  1. Training

```bash
!bash train_single_fold.sh $fold $checkpoint
```
* $fold: Fold's index is specified to model train. An integer value in the range 0 to 5.
* $checkpoint: is optional, specify the path to the checkpoint to resume training the model.


####  2. Inference

```bash
!python test.py --mode "eval" --test-set-name $testsetname --weight $weight --out-dir $outdir
```
* $testsetname: Path to the csv file containing patient ids. There are 6 csv test files corresponding for 6 folds in scripts/split/cross_val/ for a six-fold cross-validation process
* $weight: Path to weight file.
* $outdir: Path to directory where inference results will be stored.

You will see the results of FROC analysis both saved to files and printed on the screen.


##  Acknowledgement

Part of the code was adpated from [NoduleNet: Decoupled False Positive Reduction for Pulmonary Nodule Detection and Segmentation](<https://github.com/uci-cbcl/NoduleNet>)

```bash
@INPROCEEDINGS{10.1007/978-3-030-32226-7_30,
    author="Tang, Hao and Zhang, Chupeng and Xie, Xiaohui",
    editor="Shen, Dinggang and Liu, Tianming and Peters, Terry M. and Staib, Lawrence H. and Essert, Caroline and Zhou, Sean and Yap, Pew-Thian and Khan, Ali",
    title="NoduleNet: Decoupled False Positive Reduction for Pulmonary Nodule Detection and Segmentation",
    booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2019",
    year="2019",
    publisher="Springer International Publishing",
    address="Cham",
    pages="266--274",
    isbn="978-3-030-32226-7",
    doi="https://doi.org/10.1007/978-3-030-32226-7_30",
}
```
