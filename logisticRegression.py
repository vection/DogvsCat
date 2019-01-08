
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_path = "D:\\Audio_Dataset\\Animals\\annotations\\all\\train\\train"
test_path2 = "D:\\Audio_Dataset\\Animals\\annotations\\all\\train\\test"

# Parameters
image_size = 50
learning_rate = 0.001
training_epochs = 2000
batch_size = 30

# tf Graph Input
x = tf.placeholder(tf.float32, [None,image_size,image_size,1])
y = tf.placeholder(tf.float32, [None, 2])

x_ = tf.reshape(x, [-1, image_size*image_size])

# Set model weights
W = tf.Variable(tf.zeros([image_size*image_size,2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x_, W) + b) # Softmax

# cross entropy / optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cross_entropy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(image_size, image_size),
                                                             classes=['dog', 'cat'], batch_size=batch_size,
                                                             color_mode='grayscale')
    test_batches = ImageDataGenerator().flow_from_directory(test_path2, target_size=(image_size, image_size),
                                                            classes=['dog', 'cat'], batch_size=20,
                                                            color_mode='grayscale')
    # Training cycle
    for i in range(training_epochs):
        avg_cost = 0.
        total_batch = 100
        batch_xs, batch_ys = next(train_batches)
        # Run optimization op (backprop) and cost op (to get loss value)
        sess.run(train, feed_dict={x: batch_xs, y: batch_ys})
        if (i % 100 == 0):  # testing accuracy each 100 steps.
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
                acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                batch_x, batch_y = next(test_batches)
                acc_str = sess.run(acc, feed_dict={x: batch_x, y: batch_y})

                print("ON STEP: " + repr(i))
                print("Accuracy :" + repr(acc_str))
