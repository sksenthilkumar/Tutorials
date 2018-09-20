# Ref: https://www.katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-beginner

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100

# Define placeholders
with tf.variable_scope('Placeholder'):
    training_data = tf.placeholder(tf.float32, name='inputs_placeholder', shape=[None, image_size*image_size])
    labels = tf.placeholder(tf.float32, name='label_placeholder', shape=[None, labels_size])

# Variables to be tuned
with tf.variable_scope('NN'):
    W = tf.Variable(tf.truncated_normal([image_size*image_size, labels_size], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Build the network (only output layer)
with tf.variable_scope('output'):
    output = tf.add(tf.matmul(training_data, W, ), b, name="predictions")

# Define the loss function
with tf.variable_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

# Accuracy calculation
with tf.variable_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training step
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


# Run the training
# sess = tf.InteractiveSession()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(steps_number):

        # Get the next batch
        input_batch, labels_batch = mnist.train.next_batch(batch_size)
        feed_dict = {training_data: input_batch, labels: labels_batch}

        # Run the training step
        train_step.run(feed_dict=feed_dict)

        # Print the accuracy progress on the batch every 100 steps
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            print("Step %d, training batch accuracy %g %%"%(i, train_accuracy*100))

    # Evaluate on the test set
    test_accuracy = accuracy.eval(feed_dict={training_data: mnist.test.images, labels: mnist.test.labels})
    print("Test accuracy: %g %%"%(test_accuracy*100))
    saver = tf.train.Saver()
    last_chkp = saver.save(sess, 'results/graph.chkp')

