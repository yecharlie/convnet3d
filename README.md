# convnet3d
![Travis (.org) branch](https://img.shields.io/travis/yecharlie/convnet3d/master.svg?style=plastic)
![GitHub](https://img.shields.io/github/license/yecharlie/convnet3d.svg?style=plastic)

A two-stage 3D convolutional network for object detection in medical image processing...

Basically, this project is an (**non official**) reimplementation of the paper ["Automated Pulmonary Nodule Detection via 3D ConvNets with Online Sample Filtering and Hybrid-Loss Residual Learning" ](http://arxiv.org/abs/1708.03867 ).

However this project varies from what the paper states in the flowing aspects:
- Instead of using two batch in the second stage, with one batch for classifiacation, another for regresssion, this model don't perform regression for proposals (thus only one batch left in the second stage model).
- This model adopts a different input size, (15, 30, 30) on the first stage, (25, 60, 60) on the second stage, which is intened to aneurysm detection.
- The data augmentation strategy.

## Results on Aneurysms Detection
recall(%) | FPs (False Positives per Scan
--------  | -----------------------------
88.64     | 28.52

## Supported Environments
This project is developed on **keras** and **tensorflow** and tested on **python3.6**. 

## Usage
The `convnet3d/utils/kfold_dataset.py` is a demo that show how to generate dataset and train model based on the existing functionalities of this project. Overall, these steps are:

1.  Prepare the original dataset in csv file. The expected format of each line is:
    ```
    path/to/series,class,x,y,z,d,group
    ```
    where `x,y,z` are zero-based and `group` is id/tag of series, one series one id, which could be a number started from 0. 

    In addition configure the default mapping via `DATASET_CLASSES` and `DATASET_LABELS`. It may contain multiple classes and in whichever situation the background class musted be included. (`bg, 0`) 
2.  Input the original csv dataset to `makeKFold` to get k fold dataset for k-fold-cross-validation. (`kfoldCV`)
3.  Input the kfoldCV dataset to `makePatchesForDetection` to generate patches dataset for candidates screening model. Now the format of each line for patches dataset comes:
    ```
    /path/to/patch,class,x,y,z,d,path/to/series
    ```
    with `path/to/series` indicates where this patches comes from. (`cs-dataset`)
4.  Train the candidates scrrening model with one fold of patches `cs-dataset`. (`cs-model`)
5.  Input the trained `cs-model` as well as the related patches `cs-dataset` to `makePatchesForReduction` to generate patches dataset for false positive reduction model. (`fpr-dataset`)
6.  Train the false positive reduction model with `cs-model` (for transfer learning) and patches `fpr-dataset`. (`fpr-model`) 
7.  Repeat 4-6 to complete the remaining part of `kfoldCV`.

## TODO 
Add **python2.7** support.

## Notes
If you want to develop a framework exactly as the paper states, you may find all the neccessary components already shiping in. Thus you should merely rewrite a training script and a evaluation script. One suggestion is you should be careful the Out-of-Memory (OOM) error when combining the evaluation callback with training process. 
