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
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
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
WAVE_SHAPE= (1,300,300,3)
content_weight = 10
style_weight = 40
total_variation_weight = 20

# write wav params
params ={
    'nframes' : 270000,
    'nchannels':1,
    'sampwidth':2,
    'framerate':44100
}
print('content wav:'+args["wave_content"])
print('style wav:'+args["wave_style"])
if args["model"] not in MODELS.keys():
    raise AssertionError("The --model command line argument should be a key in the `MODELS` dictionary")

print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
input = K.placeholder(WAVE_SHAPE)
base_model = Network(include_top=False,weights="imagenet",input_tensor=input)
base_model.summary()
#model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)
#test: python test/xears/models/NSTshort.py -wc test/xears/data_source/test4s.wav -ws test/xears/data_source/test5s.wav -model vgg16


#初始化一个待优的占位符，这个地方待会儿实际跑起来的时候要填一个噪声
noise_wave = K.placeholder(WAVE_SHAPE)
img_nrows = int(noise_wave.shape[1])
img_ncols = int(noise_wave.shape[2])

#Let's load, reshape, and normalize our "content" wave :
content_wave,content_time = wave_utils.readWav(args["wave_content"])
content_wave = wave_utils.preprocess_wave(content_wave,img_nrows,img_ncols)
content_wave = K.variable(content_wave) #包装为Keras张量，这是一个常数的四阶张量


#Let's load, reshape and normalize our "style" wave :
style_wave,style_time = wave_utils.readWav(args["wave_style"])
style_wave = wave_utils.preprocess_wave(style_wave,img_nrows,img_ncols)
style_wave = K.variable(style_wave)#包装为Keras张量，这是一个常数的四阶张量



#将三个张量串联到一起，形成一个形如（3,img_nrows,img_ncols,3）的张量
input_tensor = K.concatenate([content_wave,
                              style_wave,
                              noise_wave], axis=0)
#print(input_tensor)
model = Network(include_top=False,weights="imagenet",input_tensor=input_tensor)
#设置Gram矩阵的计算图，首先用batch_flatten将输出的featuremap压扁，然后自己跟自己做乘法，跟我们之前说过的过程一样。注意这里的输入是某一层的representation。
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

#计算他们的Gram矩阵，然后计算两个Gram矩阵的差的二范数，除以一个归一化值，公式请参考文献
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_nrows
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#设置内容loss计算方式，以内容图片和待优化的图片的representation为输入，计算他们差的二范数，公式参考文献
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

#施加全变差正则，全变差正则用于使生成的图片更加平滑自然。
def total_variation_loss(x):
    assert K.ndim(x) == 4
    a = K.square(x[:, :img_nrows-1, :img_nrows-1, :] - x[:, 1:, :img_ncols-1, :])
    b = K.square(x[:, :img_nrows-1, :img_ncols-1, :] - x[:, :img_nrows-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))




#这是一个张量字典，建立了层名称到层输出张量的映射，通过这个玩意我们可以通过层的名字来获取其输出张量
#当然不用也行，使用model.get_layer(layer_name).output的效果也是一样的。
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
#print(outputs_dict)
#loss的值是一个浮点数，所以我们初始化一个标量张量来保存它
loss = K.variable(0.)

#layer_features就是图片在模型的block4_conv2这层的输出了，记得我们把输入做成了(3,3,nb_rows,nb_cols)这样的张量，
#0号位置对应内容图像的representation，1号是风格图像的，2号位置是待优化的图像的。计算内容loss取内容图像和待优化图像即可
layer_features = outputs_dict['block4_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
#与上面的过程类似，只是对多个层的输出作用而已，求出各个层的风格loss，相加即可。
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

#求全变差约束，加入总loss中
loss += total_variation_weight * total_variation_loss(noise_wave)

#通过K.grad获取反传梯度
grads = K.gradients(loss, noise_wave)

outputs = [loss]
#我们希望同时得到梯度和损失，所以这两个都应该是计算图的输出
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)
#编译计算图。Amazing！我们写了辣么多辣么多代码，其实都在规定输入输出的计算关系，到这里才将计算图编译了。
#这条语句以后，f_outputs就是一个可用的Keras函数，给定一个输入张量，就能获得其反传梯度了。
f_outputs = K.function([noise_wave], outputs)

def eval_loss_and_grads(x):
    # 把输入reshape层矩阵
    x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    # outs是一个长为2的tuple，0号位置是loss，1号位置是grad。我们把grad拍扁成矩阵
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        # 这个类别的事不干，专门保存损失值和梯度值
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        # 调用刚才写的那个函数同时得到梯度值和损失值，但只返回损失值，而将梯度值保存在成员变量self.grads_values中，这样这个函数就满足了func要求的条件
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        # 这个函数不用做任何计算，只需要把成员变量self.grads_values的值返回去就行了
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()
# 根据后端初始化噪声，做去均值
#x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.
x = np.random.randint(-32767, 32767, (1, img_nrows, img_ncols, 3))

for i in range(10):
    print('Start of iteration', i)
    start_time = time.time()
    # 这里用了一个奇怪的函数 fmin_l_bfgs_b更新x，我们一会再看它，这里知道它的作用是更新x就好
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    # 每次迭代完成后把输出的图片后处理一下，保存起来
    #wave = wave_utils.deprocess_wave(x.copy())
    #fname = 'wave_at_iteration_%d.wav' % i
    #wave_utils.writeWav(wave,params,fname)
    #print('wave saved as', fname)
    end_time = time.time()

print('Iteration %d completed in %ds' % (i, end_time - start_time))
