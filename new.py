import tensorflow as tf
import pandas as pd
import numpy as np

def Normalization(x): #线性函数归一化  它对原始数据进行线性变换  使结果映射到[0,1]的范围  实现对原始数据的等比缩放
    # print(x)
    x = x[0]
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

dataset = pd.read_csv('M1_201602.csv')

# x_dat = dataset.iloc[:, 3:7].values
# x_data = tf.to_float(x_dat, name='ToFloat')
# y_dat = dataset.iloc[:, -1].values
# y_data = tf.to_float(y_dat, name='ToFloat')
x_input = tf.placeholder(tf.float32,shape = [1,4])
y_output = tf.placeholder(tf.float32,shape = [1])

Weights = tf.Variable(tf.random_uniform((4,1), -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y0 = tf.matmul(x_input,Weights) + biases
y = tf.sigmoid(y0)

loss = tf.reduce_mean(tf.square(y - y_output))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init = tf.global_variables_initializer()

    for step in range(2687):
        sess.run(init)

        x_dat = dataset.iloc[step:step + 1, 3:7].values
        y_dat = dataset.iloc[step:step + 1, -1].values

        x_data = Normalization(x_dat)
        x_data = [x_data]
        print(x_data)

        sess.run(train, feed_dict={x_input: x_data, y_output: y_dat})
        print(sess.run(loss, feed_dict={x_input: x_data, y_output: y_dat}))
        print(step, sess.run(Weights), sess.run(biases))
