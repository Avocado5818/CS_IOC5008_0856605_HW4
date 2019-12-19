# Visual Recognition using Deep Learning Homework 4
## Reference
Detectron2

https://github.com/facebookresearch/detectron2

## Introduction
  This assignment is to do instance segmentation. Object detection can detect multiple different kinds of objects in the image and output individual target bounding box at the same time. Semantic Segmentation refers to the object is marked in pixels, or each pixel will  classification results. Instance segmentation combined with the above two methods can distinguish different objects in the image, and they will each have different masks.

### Installation

See [INSTALL.md](INSTALL.md).

### Train and test to make json file

Run $ python HW4_train.py

#### Setting

Model: ResNeXt-101-32x8d
pretrain weights : 

https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md

Hyper-parameters

model, pretrained weight: ResNeXt-101-32x8d

Learning Rate: 0.00025

iteration: 100000

batch size: 2

#### Result
pretrained weights: ImageNet, ResNeXt-101-32x8d

LR: 0.00025

Iteration: 120000

score: 0.55679

pretrained weights: COCO dataset, ResNeXt-101-32x8d

LR: 0.00025

Iteration: 30000

score: 0.75808

