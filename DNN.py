import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
import numpy as np
np.set_printoptions(threshold=np.inf)#To view full matrices while printing
mnist = input_data.read_data_sets(’MNIST_data’, one_hot=True)
#Changing into binary encoding
n_input = mnist.train.images.shape[1] # 28x28 = 784
n_classes = 10 # digits 0-9
def init_config(n_input, n_classes):#Setting the default graph
        ops.reset_default_graph()
        x = tf.placeholder("float32", [None, n_input], name=’x’)
        #The input set, to be replaced by batches
        y = tf.placeholder("float32", [None, n_classes], name=’y’)
        #The output training set, to be replaced by batches
        return x, y
def build_model_Llayers(x, y):
        # ops.reset_default_graph() # reset computation graph
        """
        Building a graph of 5 hidden layers
        """

        2

        # flatten the input
        #Activation used here is Relu
        print(x)
        firstlayer_dim=512
        hid = tf.layers.dense(x, units=firstlayer_dim,
        activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer()
        ,bias_initializer=tf.zeros_initializer())
        print(hid)
        hid = tf.layers.dropout(inputs=hid, rate=0.4)
        print(hid)
        hid = tf.layers.dense(hid, units=firstlayer_dim/2, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer()
        ,bias_initializer=tf.zeros_initializer())
        print(hid)
        hid = tf.layers.dense(hid, units=firstlayer_dim/4, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer() ,
        bias_initializer=tf.zeros_initializer())
        print(hid)
        hid = tf.layers.dense(hid, units=firstlayer_dim/8, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer() ,
        bias_initializer=tf.zeros_initializer())
        print(hid)
        hid = tf.layers.dense(hid, units=firstlayer_dim/16, activation=tf.nn.relu,
        kernel_initializer=tf.contrib.layers.xavier_initializer() ,
        bias_initializer=tf.zeros_initializer())
        print(hid)
        logits = tf.layers.dense(hid, units=y.shape[1], name="p",
        kernel_initializer=tf.contrib.layers.xavier_initializer() ,
        bias_initializer=tf.zeros_initializer())
        prediction = tf.one_hot(tf.cast(tf.argmax(logits, 1), tf.int32), depth=10)
        return prediction,logits
def get_loss(logits, y):
        #loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
        labels=y))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
        logits=logits))
        return loss
        def get_accuracy(pred, y):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
        #largest value in prediction vaector is the same index as the label vector
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #averaging across all the predictions in the batch
        print("accuracy %s"% accuracy)
        return accuracy
def main():
        x, y = init_config(n_input, n_classes)
        pred, logits = build_model_Llayers(x, y)
        loss = get_loss(logits, y)
        accuracy = get_accuracy(pred, y)
        train_step=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(loss)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        batchsize=1000
        for i in range(1000):
        batch = mnist.train.next_batch(batchsize) # fetch batch of size 1000
        if (i% 100) == 0:
        acc = sess.run(accuracy, feed_dict={x:batch[0], y:batch[1]})
        print(’Training accuracy at step %s: %s’ % (i, acc))
        else:
        sess.run(train_step, feed_dict={x:batch[0], y:batch[1]})
        print("Accuracy using test batch is: ")
        print(sess.run(accuracy,
        feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
        predic = sess.run(pred,
        feed_dict={x: mnist.test.images, y: mnist.test.labels})
        predic=np.asarray([np.where(r==1)[0][0] for r in predic])
        lab=np.asarray([np.where(r==1)[0][0] for r in mnist.test.labels])
        f = open("comparison.txt", "w+")
        f.write("Labels Vs. Prediction\n")
        comp=(np.c_[lab, predic])
        for line in comp:
        # print line
        f.write(" ".join(str(elem) for elem in line) + "\n")
        f.close()
main()
