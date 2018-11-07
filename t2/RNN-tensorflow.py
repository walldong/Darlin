# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 08:45:33 2018

@author: wall
"""
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8   #定义训练数据的batch大小

#神经网络结构
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
"""
在shape的一个维度上使用none可以方便的使用不同的batch大小，在训练是需要吧数据分成比较小
的batch，但是在测试时，可以一次性使用全部的数据，当数据集比较小的时候，这样比较方便测试
但是数据集比较大的时候，将大量的数据放入一个batch可能会导致内存溢出
"""
x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

#定义前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)


"""
定义损失函数和反向传播的算法
tf.clip_by_value()可以将一个张量的数值限制在一个范围内（1e-10~1.0）
tf.log完成对张量中所有元素依次求对数的过程
*表示元素直接直接相乘，tf.matmul用来矩阵相乘
y_表示真实值，y表示预测值，定义的是交叉熵损失函数
对于回归问题，最常用的损失函数是均方误差（MSE）mse = tf.reduce_mean(tf.square(y_-y))
"""
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数产生模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
Y = [[int(x1+x2<1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    
    
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000 ==0:
            total_cross_entropy=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print("after %d training steps,cross entropy on all data is %g"
                  %(i,total_cross_entropy))
    print(sess.run(w1))
    print(sess.run(w2))
