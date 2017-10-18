# Classify EMNIST Digits using Convolutional Neural Networks
EMNIST Digits dataset introduced by https://arxiv.org/abs/1702.05373v1 downloaded from https://www.nist.gov/itl/iad/image-group/emnist-dataset (Matlab format dataset). The matlab format dataset can be conveniently imported with scipy.io.loadmat.
Model was trained from scratch on EMNIST Digits training data using realtime data augmentation. All test error rates in percent.

Test error rates of this model on EMNIST Digits test data:
* 0.1699984073638916 test error rate with ensemble of 10 CNNs after 240 epochs (dropout 0.3)
* 0.1999974250793457 test error rate for best single model after 241 epochs (dropout 0.3)
