# Residual Network Design

## Description
A Residual Network Design with <5M trainable parameters. We achieved an accuracy of 96.04% on CIFAR-10 dataset by using best-suited hyperparameters and multiple training strategies like data normalization, data augmentation, optimizers, gradient clipping, etc.

## Requirements
This projects requires following external libraries:
- **torch**
- **NumPy**
- **torchvision**
- **matplotlib**

To install all the dependencies, execute: `pip install -r requirements.txt`

## Execution
### Description of files submitted
- project1_model.pt : Trained parameters/weights for our final model.
- project1_model.py : ResNet architecture used.
- train.py : place holder
- eval.py : place holder
- config.yaml : place holder

### Training
To train the model, execute: `python train.py --config config.yaml`

### Evaluation
To evaluate using the trained weights stored in project1_model.pt, execute: `python eval.py --trained_params project1_model.pt`

## Hyperparameters
| Parameter                    | Our Model       |
| ---------------------------- | --------------- |
|number of residual layers     |3                |
|number of residual blocks | [4, 3, 3]| 
|convolutional kernel sizes |[3, 3, 3] |
|shortcut kernel sizes |[1, 1, 1] |
|number of channels |64 |
|average pool kernel size |8|
|batch normalization |True |
|dropout |0 |
|squeeze and excitation |True|
|gradient clip |0.1|
|data augmentation |True|
|data normalization |True|
|lookahead |True |
|optimizer |SGD|
|learning rate (lr)| 0.1|
|lr scheduler |CosineAnnealingLR|
|weight decay |0.0005|
|batch size |128 |
|number of workers |16|
|Total number of Parameters| 4,697,742|
