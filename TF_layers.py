import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = np.array(pd.read_csv("wine.data.csv"))

X_data = data[:, 1:]
X_data = preprocessing.scale(X_data)
Y_data = data[:, 0]

iterations = 1000

tf.reset_default_graph()
tf.set_random_seed(1)
X = tf.placeholder(tf.float32, [None, 13])
Y = tf.placeholder(tf.float32, [None, 3])
#
Z1 = tf.contrib.layers.fully_connected(X, 20, activation_fn=tf.nn.relu)
Z2 = tf.contrib.layers.fully_connected(Z1, 5, activation_fn=tf.nn.relu)
Z3 = tf.contrib.layers.fully_connected(Z2, 3, activation_fn=None)
A3 = tf.nn.softmax(Z3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
objective = optimizer.minimize(cost)

init = tf.global_variables_initializer()
one_hot = tf.one_hot(Y_data, 3, axis=1)

cost_array = np.zeros(iterations)
with tf.Session() as sess:
    sess.run(init)
    Y_data = sess.run(one_hot)
    for epoch in range(iterations):
        _, temp_cost, y_cap = sess.run([objective, cost, A3], feed_dict={X: X_data, Y: Y_data})
        cost_array[epoch] = temp_cost
        # if epoch % 100 == 0:
        #     print(y_cap)

    predict_op = tf.argmax(Z3, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy)
    train_accuracy = accuracy.eval({X: X_data, Y: Y_data})
    print("Train Accuracy:", train_accuracy)

plt.ylabel('cost')
plt.xlabel('iterations')
# plt.plot([i for i in range(iterations)], y)
plt.plot(range(iterations), cost_array)




