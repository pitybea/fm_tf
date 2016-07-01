import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

print mnist.train.num_examples, mnist.test.num_examples

#model definition: 1 / (1 + exp(-M)), and here M = w * x + b + sum(<v_i, v_j> x_i * x_j)

feature_dim = 784
class_num = 10
rank_num = 10

b = tf.Variable(tf.zeros([class_num]))
w = tf.Variable(tf.zeros([feature_dim, class_num]))
v = tf.Variable(tf.random_uniform([class_num, feature_dim, rank_num], -0.005, 0.005))

x = tf.placeholder(tf.float32, shape = [None, feature_dim])
y = tf.placeholder(tf.float32, shape = [None, 10])


lr_part = tf.matmul(x, w) + b

fm_part = []

for i in range(class_num):
    v_mat = tf.matmul(x, v[i,:,:])
    v_mat_square = tf.matmul(tf.pow(x, 2), tf.pow(v[i,:,:], 2))
    fm = tf.reshape(tf.reduce_sum(tf.pow(v_mat, 2) - v_mat_square, 1), [-1,1])/ 2.0
    fm_part.append(fm)

M =  lr_part + tf.transpose(tf.pack(fm_part)[:,:,0])

prob = tf.nn.softmax(M)
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prob)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)


train_rounds = 55
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for rnd in range(train_rounds):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)     
            _, c = sess.run([train, loss], feed_dict = {x : batch_xs, y : batch_ys})
            avg_loss += c / total_batch
            #print ll.shape
            #print ff.shape
            #raw_input()
        print 'round %d, %f' % (rnd + 1, avg_loss)

        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
