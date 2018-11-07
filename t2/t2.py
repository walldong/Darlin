# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:01:10 2018

@author: wall
"""

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
"""
tf.Variable()是tensorflow的变量声明函数，可以设为随机数，常数，或者通过其他变量计算的到的数，
tf.random_normal([2,3],stddev=1)会产生一个2×3的矩阵，矩阵中的元素是均值为0，标准差为1的随机数，
可以通过参数mean来指定平均值，默认为0。
seed参数指定了随机种子，可以保证每次输出的值一样
"""
x = tf.placeholder(tf.float32, shape=(1,2), name='input')
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
"""
placeholder机制用于提供输入数据，placeholder相当于定义了一个位置，这个位置中的数据在程序运行时再
指定，这样在程序中就不需要生成大量的常量来提供输入数据，而只需要将数据通过placeholder出入tensorflow计算图
tf.matmul()实现了矩阵乘法的功能，这样可以实现神经网络的前向传播过程
"""
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
"""
tf.global_variables_initializer()函数可以实现初始化所有变量的过程，
前面的w1,w2只是定义了，并没有被赋值，所以需要在session会话中对变量进行初始化，
tf.global_variables_initializer函数会自动处理变量之间的依赖关系
"""
print (sess.run(y,feed_dict={x:[[0.7,0.9]]}))

#feed_dict是一个字典，用来指定x的取值，在字典中需要给出每个用到的placeholder的取值