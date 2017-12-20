# -*- coding: UTF-8 -*-
#build Neural Style Transfer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.models import Model
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--wave", required=True,
help="path to the input wave")
ap.add_argument("-model", "--model", type=str, default="vgg19",
help="name of pre-trained network to use")
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception, # TensorFlow ONLY
    "resnet": ResNet50
}

if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
base_model = Network(weights="imagenet")
base_model.summary()
#model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

