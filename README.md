# open cv code snippets are taken from pyimagesearch

# opencvexamples
sample code snippets - opencv
=============================

Histograms:
==========
https://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/

Searchengine
============
https://www.pyimagesearch.com/2014/01/27/hobbits-and-histograms-a-how-to-guide-to-building-your-first-image-search-engine-in-python/#pyi-pyimagesearch-plus-pricing-modal

knn.py
======
Simple knn on cats and dogs images. Sampled 2000 images of dogs and cats from kaggle.

python knn.py -d "data\dir" -k 5

keras_mnist.py (MNIST classification using keras)
==============
python .\keras_mnist.py --outpath ..\output\keras_mnist.png

keras_cifar10.py  (CIFAR classification using ANN)
==============
downloads around 170 MB of images
python .\keras_cifar10.py --output ..\output\keras_cifar10.png

convolution.py
=============
python .\convolution.py --image ..\testdata\jp.jpeg

shallownet
==========
just an example to compose a CNN - shallownet_animals.py and shallownet_cifar10.py implement shallownet

Save and load model 
==================
python shallownet_train.py --dataset ..\data --model ..\output\shallownet_weights.hdf5
python shallownet_load.py --dataset ..\data --model ..\output\shallownet_weights.hdf5

mini vggnet(cifar 10):
===========
python minivggnet_cifar10.py --output ../output/cifar10_minivggnet_with_bn.png

cifar10_monitor.py  (monitor the CNN training)
===================
python .\cifar10_monitor.py --output ..\output\


cifar10_checkpoint_improvements (checkpointing)
===============================


Kaggle - contains my solutions to kaggle competitions