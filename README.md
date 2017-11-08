# Classify EMNIST Digits using Convolutional Neural Networks
EMNIST Digits dataset introduced by https://arxiv.org/abs/1702.05373v1 downloaded from https://www.nist.gov/itl/iad/image-group/emnist-dataset (Matlab format dataset). The matlab format dataset can be conveniently imported with scipy.io.loadmat.
Model was trained from scratch on EMNIST Digits training data using realtime data augmentation. 

![accuracies](plot_history/accuracy.png)![ensemble accuracies](plot_history/accuracy_ensembles.png)

# Test error rates of this model on EMNIST Digits test data:
## Dropout 0.2
* 0.1675% test error rate with ensemble of 10 CNNs after 256 epochs
* 0.20% test error rate for best single model after 128 epochs

## Dropout 0.3
* 0.17% test error rate with ensemble of 10 CNNs after 240 epochs
* 0.20% test error rate for best single model after 241 epochs

## Dropout 0.4
* 0.1675% test error rate with ensemble of 10 CNNs after 288 epochs
* 0.21% test error rate for best single model after 272 epochs

## Dropout 0.5
* 0.1825% test error rate with ensemble of 10 CNNs after 208 epochs
* 0.21% test error rate for best single model after 144 epochs
