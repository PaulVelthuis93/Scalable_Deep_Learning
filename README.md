# Project AM-FED dataset

## Abstract
This project is about detecting facial expressions from videos. It involves a dataset with 242 videos with labelled facial expressions. For this project we use the dataset: Afectiva-MIT Facial Expression Dataset (AM-FED): Naturalistic and Spontaneous Facial Expressions Collected In-the-Wild Paper. In this dataset there is labelled data available about gestures people make during the video. In this project we analyse if people are smiling or not based on the data from this dataset.

## What the code does
Extract frames/images from videos and then analyse them on emotions, for example happy and not happy, sad and angry. 

## Requirements
To analyse the videos for emotions we will perform deep learning. It provides the possibility to detect smiles. In order to perform Deep learning there are several requirements which are described below. 	 	 	
Tensorflow (latest version 0.12) is an open source library released by Google. Tensorflow provides the ability to perform deep learning on images. It can make use of the GPU to perform image processing. With image processing it is possible to recognize images, and thus to recognize gestures.
Tensorflow requires Python 2.7 or higher to run. Since Python 2.7 is widely supported we decided to use Python 2.7.
Setup Python 2.7 Linux:
On Linux python is often pre-installed, with the following command, you can see which version you have:
```bash
$ python --version  #When reading such code you always type the text after the $ in your terminal 
```
If Python is not installed: 
```bash
$ sudo apt-get install python2.7
```
To install Tensorflow there are installation guidelines to be found in the following link:
https://www.tensorflow.org/get_started/os_setup
In order to install Tensorflow the pip package management system of python needs to be installed. If not then there you can install it on linux with:
```bash
$ sudo apt-get install python-pip python-dev
```
Tensorflow performs better on the GPU than the CPU. To install version 0.12 the current latest version on a computer or laptop without GPU you type:
```bash
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0rc1-cp27-none-linux_x86_64.whl
```
If you have a GPU on your computer or laptop you install with the following command:
```bash
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.0rc1-cp27-none-linux_x86_64.whl
```
Tensorflow can only operate on image data, for this reason we need to extract the images from the videos. For this we use OpenCV (Version 2.4.9.1), with which it is possible to capture the image from a certain time in the video. To install OpenCV type:
```bash
$ sudo apt-get install python-opencv
```
We use the Python Image Library (PIL) to extract the information from an image. To install the PIL library:
```bash
$ sudo apt-get install python-imaging
```
We make a numpy array of the pixels to analyse the images , so it is also required. Usage for numpy is available at the following link: https://www.scipy.org/install.html
Numpy can be installed with the following command:
```bash
$ sudo apt-get install python-numpy
```
## Steps taken
1. To start with analysing the videos, we first take the image from the videos. The videos contain a CSV file which contains information about which gestures were shown at which moment in time. So we can extract labelled images with for example the information whether the person is smiling or not. Thus our first step is to extract the labeled images from the videos.
2. Next, analyses of the images was rendered using Tensorflow. In the coding section, we explain more about how our Tensorflow algorithm works. 



