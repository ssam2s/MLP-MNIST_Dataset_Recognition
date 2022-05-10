import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

def one_hot(lables, targets=10):
    samples = len(lables)
    out = np.zeros((samples, targets))
    out[range(samples), lables] = 1
    return out

mnist = np.load('mnist.npz')
x_train = mnist['x_train'] / 255
y_train = one_hot(mnist['y_train'] , 10)
x_test = mnist['x_test'] / 255
y_test = one_hot(mnist['y_test'], 10)

learning_rate = 0.05

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input')
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='target')

w = tf.Variable(tf.truncated_normal(shape=[784, 10], stddev=0.1), name='weight')
b = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32), name='bias')
z = tf.matmul(x, w) + b
yhat = tf.nn.softmax(z)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
loss = tf.reduce_mean(loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

correct = tf.equal(tf.argmax(y, 1), tf.argmax(yhat, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

EPOCH = 5000
BATCH = 1000
steps = x_train.shape[0] // BATCH

x_train_batch = x_train.reshape(-1, BATCH, 28*28)
y_train_batch = y_train.reshape(-1, BATCH, 10) 

x_test = x_test.reshape(-1, 28*28)
y_test = y_test.reshape(-1, 10)

loss_epoch=[]
accuracy_epoch = []

loss_test = []
accuracy_test = []

for epoch in range(1, EPOCH+1):
    loss_batch = []
    accuracy_batch = []

    for step in range(steps):
        x_batch = x_train_batch[step]
        y_batch = y_train_batch[step]
        _, loss_, acc_ = sess.run([train, loss, accuracy], feed_dict = {x: x_batch, y: y_batch})
        print('epoch={:4d} step={:4d} loss={:12.8f} accuracy = {:6.5f}'.format(epoch, step, loss_, acc_))

        loss_batch.append(loss_)
        accuracy_batch.append(acc_)

    mean_loss = np.mean(loss_batch)
    mean_accuracy = np.mean(accuracy_batch)
    loss_epoch.append(mean_loss)
    accuracy_epoch.append(mean_accuracy)
    print('train: loss={:.8f} accuracy={:.5f}'.format(mean_loss, mean_accuracy))

    loss_, acc_ = sess.run([loss, accuracy], feed_dict = {x: x_test, y: y_test})
    loss_test.append(loss_)
    accuracy_test.append(acc_)
    print('test: loss={:.8f} accuracy={:.5f}'.format(loss_, acc_))

sess.close()

plt.plot(loss_epoch, 'r')
plt.plot(loss_test, 'b')
plt.show()

plt.plot(accuracy_epoch, 'r')
plt.plot(accuracy_test, 'b')
plt.show()