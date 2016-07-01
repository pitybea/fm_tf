import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

#model definition: 1 / (1 + exp(-M)), and here M = w * x + b + sum(<v_i, v_j> x_i * x_j)


feature_dim = 10
rank_num = 3

b = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([feature_dim, 1]))
v = tf.Variable(tf.random_uniform([feature_dim, rank_num], -0.05, 0.05))

x = tf.placeholder(tf.float32, shape = [None, feature_dim])
y = tf.placeholder(tf.float32, shape = [None])

v_mat = tf.matmul(x, v)
v_mat_square = tf.matmul(tf.pow(x, 2), tf.pow(v, 2))

lr_part = tf.matmul(x, w) + b
fm_part = tf.reshape(tf.reduce_sum(tf.pow(v_mat, 2) - v_mat_square, 1), [-1,1])/ 2.0 
M =  lr_part + fm_part

prob = tf.sigmoid(M)
loss = tf.reduce_mean(tf.log(tf.add(1.0, tf.exp(-y * M))))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

#data 
sample_size = 200

x_pos = np.random.rand(sample_size, feature_dim) - 1.1
y_pos = np.ones(sample_size)

x_neg = np.random.rand(sample_size, feature_dim)
y_neg = np.zeros(sample_size)

rand_ind = np.random.permutation(sample_size * 2)

x_data = np.concatenate([x_pos, x_neg])[rand_ind]
y_data = np.concatenate([y_pos, y_neg])[rand_ind]

#train

train_rounds = 10

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for rnd in range(train_rounds):
        for i in range(len(x_data)):
            sess.run(train, feed_dict = {x : [x_data[i]], y : [y_data[i]]})
        
        dic = {x : x_data, y : y_data}
        predicts = sess.run(prob, feed_dict = dic)

        print 'roc', roc_auc_score(y_data, predicts)
        print loss, sess.run(loss, feed_dict = dic)

