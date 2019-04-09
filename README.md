# DNN
## Simple DNN on MNIST
1 Algorithm and Python Code
A DNN has been created here for a training and testing framework using the
MNIST dataset using tensorflow. The deep neural network model created here
consists of 5 hidden layers with dimension of the first hidden layer being 512.The
subsequent hidden layers have dimensions equal to half of the preceding layer.
### 1.1 Initialization
We start out by importing the dataset from tensorflow and setting one-hot as
true, as this ensures a binary encoding of the labels to a vector of length 10. This
makes the final comparison easier. Then we define placeholders for the inputs,
which are the 28X28 pixels of every image in the dataset with an undefined size
as the batch sizes may be altered in the subsequent iterations, and the output
classes similarly with a shape of (None, 10).
### 1.2 Building the model
Next, we build our model of L layers using the x & y values and the tf.layers.dense

function. The activation function we use here is Relu. We build a neural net-
work with 5 hidden layers and dimensions as 512, 256, 128,64 and 32 as follows,

finally culminating into the output layer of dimension 10 as it represents the
classes.The weights and biases are initialized using the xavier initializer() and
zeros initializer() respectively. Xavier Initializer is known to adjust the weights
so that the activation reaches deep into the last layer of the network in contrast
to using a random initializer. We thus, compute the logits after the computation
through the 5 hidden layers. Here, logits mean the unscaled probabilities.The
prediction is then calculated by finding out the index/class with highest logit
value. this prediction is converted to a one-hot vector like the input labels.


### 1.3 Loss and Accuracy
Then we compute the loss where we use softmax to scale the probabilities into
a valid probability distribution and thus verify the loss using the logits as the

input. Finally we get the accuracy by ascertaining if the largest value in pre-
diction vector is the same index as the label vector. The mean is then returned

as the required accuracy.
### 1.4 Optimizer
For the back-propagation,Adam Optimizer is used as it gave better results in
less computation time than gradient descent.
### 1.5 Operation
The built network is trained by using random batches of 1000 samples running
the algorithm for a total of 1000 epochs.The training performance thus obtained
is 100% and the testing performance achieved is 98.11% .A final comparison of
the MNIST test labels and the network’s prediction is thus finally published in
the file attached.
