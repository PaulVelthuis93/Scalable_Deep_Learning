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



