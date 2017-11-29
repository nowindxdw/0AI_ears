#coding:utf-8
#dnn_eval.py

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


import dnn_inference
import dnn_train

#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10
INPUT_NODE,LAYER1_NODE,LAYER2_NODE,OUTPUT_NODE = dnn_inference.get_node_dims()

def evaluate(model):
    with tf.Graph().as_default() as g:
    #定义输入输出格式
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    validate_feed = {x: model.validation.images, y_: model.validation.labels}
    #计算前向传播结果，测试时不关心正则化损失的值，所以这里设为None
    y = mnist_inference.inference(x, None)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    while True:
        with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(dnn_train.MODEL_SAVE_PATH)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
        #加载模型
            saver.restore(sess, ckpt.model_checkpoint_path)
            #通过文件名得到模型保存时迭代的轮数
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

            accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
            print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
        else:
            print('No checkpoint file found')
            return
        time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets("/User/dawei/AI/DNN/data", one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
   tf.app.run()