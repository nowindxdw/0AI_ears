#coding:utf-8
#dnn_inference.py

import tensorflow as tf

##定义神经网络相关参数
def get_node_dims(layer_dims=[784,512,512,10]):
    INPUT_NODE = layer_dims[0]
    LAYER1_NODE = layer_dims[1]
    LAYER2_NODE = layer_dims[2]
    OUTPUT_NODE = layer_dims[3]
    return INPUT_NODE,LAYER1_NODE,LAYER2_NODE,OUTPUT_NODE

#通过tf.get_variable 函数来获取变量
#在训练神经网络时会创建这些变量
#在测试时会通过保存的模型，加载这些变量的取值
#可以在变量加载时将“滑动平均变量”重命名，所以可以在训练时使用变量自身，在测试时使用变量的滑动平均值
#在这个函数中也会将变量的正则化损失加入损失集合
def get_weight_variable(shape, regularizer):
    #对权重的定义，shape表示维度
    #将变量初始化为满足正太分布的随机值，但如果随机出来的值偏离平均值超过2个标准差，那么这个数将会被重新随机
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    #将当前变量的正则损失加入名字为losses的集合
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    INPUT_NODE,LAYER1_NODE,LAYER2_NODE,OUTPUT_NODE = get_node_dims()
    #声明第一层神经网络的变量并完成前向传播的过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    #声明第二层神经网络的变量并完成前向传播的过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, LAYER2_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    #声明第三层神经网络的变量并完成前向传播的过程
    with tf.variable_scope('layer3'):
            weights = get_weight_variable([LAYER2_NODE, OUTPUT_NODE], regularizer)
            biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer3 = tf.matmul(layer2, weights) + biases
    return layer3