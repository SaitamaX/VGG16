import tensorflow as tf
import numpy as np
import pdb
from datetime import datetime
from VGG16 import *
import os
import cv2


def test(path):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, 17)
    score = tf.nn.softmax(output)
    f_cls = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # 训练好的模型位置
    saver.restore(sess, './model/model.ckpt-49999')
    for i in os.listdir(path):
        imgpath = os.path.join(path, i)
        im = cv2.imread(imgpath)
        cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('input_image', im)
        im = cv2.resize(im, (224, 224))  # * (1. / 255)
        im = np.expand_dims(im, axis=0)
        # 测试时，keep_prob设置为1.0
        pred, _score = sess.run([f_cls, score], feed_dict={x: im, keep_prob: 1.0})
        prob = round(np.max(_score), 4)
        print("{} flowers class is: {}, score: {}".format(i, int(pred), prob))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    sess.close()


if __name__ == '__main__':
    # 测试图片保存在文件夹中了，图片前面数字为所属类别
    path = './test'
    test(path)
