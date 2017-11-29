#coding:utf-8
#dnn_train.py

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#加载 dnn_inference.py 中定义的常量和前向传播的函数
import dnn_inference

#配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 9001
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH = os.path.dirname(__file__)+'/DNN/output/model'
MODEL_NAME = "model.ckpt"

INPUT_NODE,LAYER1_NODE,LAYER2_NODE,OUTPUT_NODE = dnn_inference.get_node_dims()


def train(data_set):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
    #返回regularizer函数，L2正则化项的值
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    #使用dnn_inference.py中定义的前向传播过程
    y=dnn_inference.inference(x,regularizer)
    #定义step为0
    global_step = tf.Variable(0, trainable=False)

    #滑动平均,由衰减率和步数确定
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #可训练参数的集合
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #交叉熵损失 函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = tf.argmax(y_, 1))
    #交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #总损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #学习率(衰减)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, data_set.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    #定义了反向传播的优化方法，之后通过sess.run(train_step)就可以对所有GraphKeys.TRAINABLE_VARIABLES集合中的变量进行优化，似的当前batch下损失函数更小
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #更新参数
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    #初始会话，并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = data_set.train.next_batch(BATCH_SIZE)
            op, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print ("After %d training step(s), loss on training batch is %g." % (step, loss_value))

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("/User/dawei/AI/DNN/data", one_hot=True)
    train(mnist)

#if __name__ == '__main__':
#   tf.app.run()
main()