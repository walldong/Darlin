# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:31:14 2018

@author: wall
"""
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32, shape=(3,2), name='input')
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)


sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)


print (sess.run(y,feed_dict={x:[[0.7,0.9],[0.4,0.1],[0.5,0.8]]}))
"""
由于前面x的值为了3×2的矩阵，所以在运行前向传播过程是需要提供3个样例数据
所以此时可以得到3个样例数据的前向传播的输出
"""