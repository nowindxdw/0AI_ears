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
import tensorflow as tf

import sys
import os
import_dir = \
os.path.join(os.path.join(os.path.dirname(__file__),os.pardir),'data_utils')
sys.path.insert(0,import_dir)
file_name = 'wave_utils'
wave_utils = __import__(file_name)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-wc", "--wave_content", required=True,
help="path to the input content wave")
ap.add_argument("-ws", "--wave_style", required=True,
help="path to the input style wave")
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
print('content wav:'+args["wave_content"])
print('style wav:'+args["wave_style"])
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
base_model = Network(weights="imagenet",include_top=False)

base_model.summary()
#model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
#test: python test/xears/models/NST.py -wc test/xears/data_source/test20.wav -ws test/xears/data_source/test30.wav -model vgg16

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(tf.transpose(a_C,perm=[0,3,1,2]),[n_C, n_H * n_W])
    a_G_unrolled = tf.reshape(tf.transpose(a_G,perm=[0,3,1,2]),[n_C, n_H * n_W])

    # compute the cost with tensorflow (≈1 line)
    J_content =  (1 / (4 * n_H * n_W * n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))

    return J_content
"""
#validate compute_content_cost
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))

#J_content = 6.76559
"""

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A,tf.transpose(A))

    return GA
"""
#validate gram_matrix
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)

    print("GA = " + str(GA.eval()))

#GA = [[  6.42230511  -4.42912197  -2.09668207]
# [ -4.42912197  19.46583748  19.56387138]
# [ -2.09668207  19.56387138  20.6864624 ]]
"""

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S,perm=[0,3,1,2]),[n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G,perm=[0,3,1,2]),[n_C, n_H * n_W])

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    # Computing the loss
    J_style_layer = (1 / (4 * (n_C * n_H * n_W)**2))*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))

    return J_style_layer

"""
#validate compute_layer_style_cost
tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)

    print("J_style_layer = " + str(J_style_layer.eval()))
#J_style_layer = 9.19028
"""

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        #out = model[layer_name]
        out = Model(input=model.input, output=model.get_layer(layer_name).output)
        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha * J_content + beta * J_style

    return J

"""
#validate total_cost
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))
#J = 35.34667875478276
"""
# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

#Let's load, reshape, and normalize our "content" wave :
content_wave,content_time = wave_utils.readWav(args["wave_content"])
#print(content_wave.shape)
#(1, 1350720)

#Let's load, reshape and normalize our "style" wave :
style_wave,style_time = wave_utils.readWav(args["wave_style"])

#wave_utils.drawWave(content_wave,content_time)
#wave_utils.drawWave(style_wave,style_time)


