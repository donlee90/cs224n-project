import tensorflow as tf
import numpy as np
from model import Model
import time
from utils.general_utils import get_minibatches

class Config(object): #BK: (object) means that Config class inherits from "object" class, and in practice I can replace it with ().
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    seq_length = 10
    embed_size = 5
    batch_size = 50
    n_epochs = 10
    lr = 1e-4
    cell_size = 2000
    clip_gradients = False
    max_grad_norm = 1e3

class Seq2seq_autoencoder(Model): #For my own benefit: (Model) means SoftmaxModel class inherits from Model class defined in model.py. Therefore, it can use self.build() for e.g.
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
        ### 
        input_placeholder = tf.placeholder(tf.float32,(None, self.config.seq_length, self.config.embed_size))
        labels_placeholder = tf.placeholder(tf.float32,(None, self.config.seq_length, self.config.embed_size))
        self.input_placeholder = input_placeholder
        self.labels_placeholder = labels_placeholder
        ### 

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### 
        feed_dict = {self.input_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        ### 
        return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. 


        """
       	enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.cell_size) #temp: can use different types of rnn
        enc_input = self.input_placeholder
        with tf.variable_scope("encoder"):
        	_, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_input, None, tf.zeros((tf.shape(enc_input)[0], enc_cell.state_size),tf.float32)) #None represents the argument 'sequence_length', which is a tensor of shape [batch_size], which specifies the length of the sequence for each element of the batch. The fourth arg is initial state.

        dec_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.cell_size) #temp: can use different types of rnn
        dec_cell = tf.nn.rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size)

        #Creates a decoder input, for which we append the zero vector at the beginning of each sequence, which serves as the "GO" token.
        unpacked_enc_input = tf.unpack(enc_input, axis=1)
        unpacked_dec_input = [tf.zeros_like(unpacked_enc_input[0])] + unpacked_enc_input[:-1]
        dec_input = tf.pack(unpacked_dec_input, axis=1)
        
        with tf.variable_scope("decoder"):
       		dec_output, _ = tf.nn.dynamic_rnn(dec_cell, dec_input, None, enc_state)	
       	pred = dec_output

        return pred

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        """
        ### 
        loss = tf.nn.l2_loss(pred-self.labels_placeholder)
        ### 
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### 
        #train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)

        grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables()) #BK: this is a list of (gradient,variable) pairs. Check if this works.
        gradients = [pair[0] for pair in grads_and_vars]

        if self.config.clip_gradients:
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)

        self.grad_norm = tf.global_norm(gradients)
        grads_and_vars = [(gradients[idx],pair[1]) for idx, pair in enumerate(grads_and_vars)]
        train_op = optimizer.apply_gradients(grads_and_vars)  

        ###
        assert self.grad_norm is not None, "grad_norm was not set properly!"
        return train_op

    def run_epoch(self, sess, inputs, labels):
        """Runs an epoch of training.

        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size): #For my own benefit: get_minibatches randomly selects as many as batch_size from the total training samples. This implements SGD.
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches

    def fit(self, sess, inputs, labels):
        """Fit model on provided data.

        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch+1, average_loss, duration)
            losses.append(average_loss)
        return losses

    def __init__(self, config):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()

def generate_sequence_data(train_size, seq_length, embed_size, mean=0, std=1):
	"""Each element is drawn from i.i.d Gaussians.
	Note that train_size, the size of the entire training set, is in general different from and much larger than batch_size, which refers to the size of minibatches.
	"""
	return std*np.random.randn(train_size, seq_length, embed_size) + mean 

def do_seq2seq_prediction():
	config = Config()
	#Generate a training data
	np.random.seed(0)
	inputs = generate_sequence_data(10*config.batch_size, config.seq_length, config.embed_size, mean=0, std=1)
	labels = inputs

	#Create and train a seq2seq autoencoder
	with tf.Graph().as_default():
		model = Seq2seq_autoencoder(config)
		init = tf.global_variables_initializer()

		with tf.Session() as sess:
			sess.run(init)
			model.fit(sess, inputs, labels)

			#test
			np.random.seed(1)
			test_inputs = generate_sequence_data(config.batch_size, config.seq_length, config.embed_size, mean=0, std=1)
			test_labels = test_inputs
			print model.loss_on_batch(sess, test_inputs, test_labels)

			np.random.seed(2)
			test_inputs = generate_sequence_data(config.batch_size, config.seq_length, config.embed_size, mean=0, std=1)
			test_labels = test_inputs
			print model.loss_on_batch(sess, test_inputs, test_labels)

			np.random.seed(3)
			test_inputs = generate_sequence_data(config.batch_size, config.seq_length, config.embed_size, mean=0, std=1)
			test_labels = test_inputs
			print model.loss_on_batch(sess, test_inputs, test_labels)

if __name__ == "__main__":
	do_seq2seq_prediction()

