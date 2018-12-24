## Wrriten by: ##
# Aviv Harazi
# Maor Bakshi
# Yoni Tseva

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Source paths, each folder contains sub folders of the classes.
# for example folder train has dog and cat sub folders containing images.
train_path = "D:\\Audio_Dataset\\Animals\\annotations\\all\\train\\train"
test_path2 = "D:\\Audio_Dataset\\Animals\\annotations\\all\\train\\test"

# Image size - (200,200)
image_size = 200

# Setting weights .
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# Setting bias.
def init_bias(shape):
    init_bias_val = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_val)

# Concolution 2d .
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Max pooling
def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Setting conolutional layer .
def conv_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

# Setting fully connected layer.
def full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# CNN Model
# Placeholders


x = tf.placeholder(tf.float32, shape=[None, image_size , image_size, 1]) # one dimension for grey scale.
y_true = tf.placeholder(tf.float32, shape=[None, 2])

x_image = tf.reshape(x, [-1, image_size, image_size, 1])

# Layer 1
convo_1 = conv_layer(x_image, shape=[3, 3, 1, 32])
convo_1_pooling = max_pool_2by2(convo_1)

# Layer 2
convo_2 = conv_layer(convo_1_pooling, shape=[3, 3, 32, 64])
convo_2_pooling = max_pool_2by2(convo_2)

# Layer 3
new_image_size = image_size / 4
convo_2_flat = tf.reshape(convo_2_pooling, [-1, int(new_image_size * new_image_size * 64)])
full_layer_one = tf.nn.relu(full_layer(convo_2_flat, 1024))

# Layer 4
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

y_pred = full_layer(full_one_dropout, 2)

# Softmax
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)


saver = tf.train.Saver()
init = tf.global_variables_initializer()
steps = 3000
with tf.Session() as sess:
    sess.run(init)

    # Prepare batches
    batch_size = 50
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(image_size, image_size), classes=['dog', 'cat'], batch_size=batch_size,color_mode='grayscale')
    test_batches = ImageDataGenerator().flow_from_directory(test_path2, target_size=(image_size, image_size),classes=['dog', 'cat'], batch_size=20,color_mode='grayscale')

    print("Training has been started...")

    for i in range(steps):
        # feeding data
        batch_x, batch_y = next(train_batches)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

        if (i % 100 == 0) : # testing accuracy each 100 steps.
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            batch_x, batch_y = next(test_batches)
            acc_str = sess.run(acc, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 1.0})

            print("ON STEP: " + repr(i))
            print("Accuracy :" + repr(acc_str))

    # Saving the model
    saver.save(sess, './dog_vs_cat.ckpt')

