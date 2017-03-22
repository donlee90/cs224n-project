import time
import tensorflow as tf
from model import Model
from utils.general_utils import get_minibatches


class LogisticConfig(object): #For my own benefit: (object) means that Config class inherits from "object" class, and in practice I can replace it with ().
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_samples = 1024
    input_size = 100
    n_classes = 5
    batch_size = 64 #For my own benefit: batch_size really means mini_batch_size. The total number of training samples (which is sometimes called "batch_size" is n_samples above.)
    n_epochs = 50
    lr = 1e-3
    label_max = 5 #label is a real continuous variable in [0,label_max].


class LogisticModel(Model): #For my own benefit: (Model) means SoftmaxModel class inherits from Model class defined in model.py. Therefore, it can use self.build() for e.g.
    """Implements a Softmax classifier with cross-entropy loss."""

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.

        These placeholders are used as inputs by the rest of the model building
        and will be fed data during training.

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of shape
                                              (batch_size, n_features), type tf.float32
        labels_placeholder: Labels placeholder tensor of shape
                                              (batch_size, n_classes), type tf.int32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
        """
        ### YOUR CODE HERE
        input_placeholder = tf.placeholder(tf.float32,(None, self.config.input_size)) 
        labels_placeholder = tf.placeholder(tf.float32,(None,))
        self.input_placeholder = input_placeholder
        self.labels_placeholder = labels_placeholder
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If label_batch is None, then no labels are added to feed_dict.

        Hint: The keys for the feed_dict should be the placeholder
                tensors created in add_placeholders.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE HERE
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        ### END YOUR CODE
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:

        y = softmax(Wx + b)

        Hint: Make sure to create tf.Variables as needed.
        Hint: For this simple use-case, it's sufficient to initialize both weights W
                    and biases b with zeros.

        Args:
            input_data: A tensor of shape (batch_size, n_features).
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        ### YOUR CODE HERE
        x = self.input_placeholder
        W = tf.Variable(tf.zeros((self.config.input_size,1)))
        b = tf.Variable(tf.zeros((1,)))
        pred = tf.matmul(x,W) + b #(None, 1)
        pred = tf.reshape(pred, (-1,)) #(None)
        pred = tf.sigmoid(pred)
        ### END YOUR CODE
        return pred

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        Hint: Use the cross_entropy_loss function we defined. This should be a very
                    short function.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        loss = tf.nn.l2_loss(pred-self.labels_placeholder)
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer to get an optimizer object.
                    Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE
        #train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def run_epoch(self, sess, examples):
        """Runs an epoch of training.

        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches(examples, self.config.batch_size): #For my own benefit: get_minibatches randomly selects as many as batch_size of the total training samples. This implements SGD.
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches

    def evaluate(self, sess, examples):
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches(examples, self.config.batch_size):
            n_minibatches += 1
            total_loss += self.loss_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches

    def fit(self, sess, train, dev, test):
        """Fit model on provided data.

        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        best_score = float('inf')
        test_loss = float('inf')
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            train_loss = self.run_epoch(sess, train)
            dev_loss = self.evaluate(sess, dev)
            duration = time.time() - start_time
            print 'Epoch {:}: train loss = {:.2f} dev_loss = {:.2f} ({:.3f} sec)'.format(epoch, train_loss, dev_loss, duration)
            losses.append(train_loss)

            if dev_loss < best_score:
                best_score = dev_loss
                print 'New best score! Evaluating on test set...'
                test_loss = self.evaluate(sess, test)
                print 'test loss = {:.2f}'.format(test_loss)
        return losses, test_loss

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()
