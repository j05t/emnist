# Classify EMNIST Digits using Convolutional Neural Networks
EMNIST Digits dataset introduced by https://arxiv.org/abs/1702.05373v1 downloaded from https://www.nist.gov/itl/iad/image-group/emnist-dataset (Matlab format dataset). The matlab format dataset can be conveniently imported with scipy.io.loadmat.
Model was trained from scratch on EMNIST Digits training data using realtime data augmentation. 

All test error rates in percent. Source code resides in ![emnist.ipynb](emnist.ipynb), detailed results can be viewed at ![plot_models](plot_history/plot_csv.ipynb). Best model weights have been uploaded.

![accuracies](plot_history/accuracy.png)![ensemble accuracies](plot_history/accuracy_ensembles.png)

# Test error rates of this model on EMNIST Digits test data:

## Automatic learning rate adjustment with Adaptive Moment Estimation
### Dropout 0.2
* 0.1675% test error rate with ensemble of 10 CNNs after 256 epochs
* 0.20% test error rate for best single model after 128 epochs

### Dropout 0.3
* 0.17% test error rate with ensemble of 10 CNNs after 240 epochs
* 0.20% test error rate for best single model after 241 epochs

### Dropout 0.4
* 0.1675% test error rate with ensemble of 10 CNNs after 288 epochs
* 0.21% test error rate for best single model after 272 epochs

### Dropout 0.5
* 0.1825% test error rate with ensemble of 10 CNNs after 208 epochs
* 0.21% test error rate for best single model after 144 epochs

## Manually set learning rates:
The most promising previously trained models have been loaded, further training was done with a fixed learning rate.
### Dropout 0.2
Loaded 256 epoch models, learning rate set to 0.00001
* 0.1600027084350586% test error rate with ensemble of 10 CNNs at 267 epochs
* 0.17750263214111328% test error rate for best single model at 262 epochs (262epochs_weights_model_8.pkl)
