# convnet3d
A two stage 3d convolutional network for object detection in medical image processing...

Basically, this project is an (**non official**) reimplementation of the paper ["Automated Pulmonary Nodule Detection via 3D ConvNets with Online Sample Filtering and Hybrid-Loss Residual Learning" ](http://arxiv.org/abs/1708.03867 ).

However this project vary from what the paper states in the flowing aspects:
- Instead of using two batch in the second stage, with one batch for classifiacation, another for regresssion, this model directly don't performa regression for proposals.
- This model adaop a different input size, (15, 30, 30) on the first stage, (25, 60, 60) on the second stage, which is intened to aneurysm detection.
- The data augmentation strategy.

If you want to develop a framework exactly as the paper states, you may find all the neccessary components already shiping in. Thus you should merely rewrite a training script and a evaluation script. One suggestion is you should be careful the OOM (Out of Memory) error when combining the evaluation callback with training process. 

This project is originally developed for aneurysm detection. And some of the codes may be coupled with aneurysn dataset feature. I will try to untie this correlation for a general use of this framework. 

This project is developed on **keras** and **tensorflow** and tested on **python3.6**. In the future, It may add **python2.7** support.
