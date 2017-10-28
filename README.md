# Classify EMNIST Digits using Convolutional Neural Networks
EMNIST Digits dataset introduced by https://arxiv.org/abs/1702.05373v1 downloaded from https://www.nist.gov/itl/iad/image-group/emnist-dataset (Matlab format dataset). The matlab format dataset can be conveniently imported with scipy.io.loadmat.
Model was trained from scratch on EMNIST Digits training data using realtime data augmentation. 

# Test error rates of this model on EMNIST Digits test data:
All test error rates in percent.
## Dropout 0.2
* 0.16750097274780273 test error rate with ensemble of 10 CNNs after 256 epochs
* 0.199997425079346 test error rate for best single model after 128 epochs

## Dropout 0.3
* 0.1699984073638916 test error rate with ensemble of 10 CNNs after 240 epochs
* 0.1999974250793457 test error rate for best single model after 241 epochs

## Dropout 0.4
* 0.167500972747803 test error rate with ensemble of 10 CNNs after 288 epochs
* 0.209999084472656 test error rate for best single model after 272 epochs
