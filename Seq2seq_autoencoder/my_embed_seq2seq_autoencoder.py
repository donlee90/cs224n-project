import tensorflow as tf
import numpy as np
from model import Model
import time
from utils.general_utils import minibatches

class Config(object): #BK: (object) means that Config class inherits from "object" class, and in practice I can replace it with ().
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    seq_length = 10 # max_length
    embed_size = 5
    batch_size = 50
    n_tokens = 7
    n_epochs = 20
    lr = 1e-3
    cell_size = 200
    clip_gradients = False
    max_grad_norm = 1e3
    padding_int = 0


class Seq2seq_autoencoder(Model): #For my own benefit: (Model) means SoftmaxModel class inherits from Model class defined in model.py. Therefore, it can use self.build() for e.g.
    """Implements a Softmax classifier with cross-entropy loss."""

    def pad_sequences(self, data, max_length):
        """Ensures each input-output seqeunce pair in @data is of length
        @max_length by padding it with zeros and truncating the rest of the
        sequence.

        TODO: In the code below, for every sentence, labels pair in @data,
        (a) create a new sentence which appends zero feature vectors until
        the sentence is of length @max_length. If the sentence is longer
        than @max_length, simply truncate the sentence to be @max_length
        long.
        (b) create a new label sequence similarly.
        (c) create a _masking_ sequence that has a True wherever there was a
        token in the original sequence, and a False for every padded input.

        Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
        0, 0], and max_length = 5, we would construct
            - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
            - a new label seqeunce: [1, 0, 0, 4, 4], and
            - a masking seqeunce: [True, True, True, False, False].

        Args:
            data: is a list of (sentence, labels) tuples. @sentence is a list
                containing the words in the sentence and @label is a list of
                output labels. Each word is itself a list of
                @n_features features. For example, the sentence "Chris
                Manning is amazing" and labels "PER PER O O" would become
                ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
                the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
                is the list of labels. 
            max_length: the desired length for all input/output sequences.
        Returns:
            a new list of data points of the structure (sentence', labels', mask).
            Each of sentence', labels' and mask are of length @max_length.
            See the example above for more details.
        """
        ret = []
        # padding_int is the index or integer used for padding. 
        padding_int = self.config.padding_int

        for sentence, labels in data:
            ### YOUR CODE HERE (~4-6 lines)
            mask = []
            padded_sentence = []
            padded_labels = []
            for i in xrange(max_length):
                if i < len(sentence):
                    mask.append(True)
                    padded_sentence.append(sentence[i])
                    padded_labels.append(labels[i])
                else:
                    mask.append(False)
                    padded_sentence.append(padding_int)
                    padded_labels.append(padding_int)

            ret.append((padded_sentence, padded_labels, mask))
            ### END YOUR CODE ###
        return ret

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
        input_placeholder = tf.placeholder(tf.int32,(None, self.config.seq_length))
        labels_placeholder = tf.placeholder(tf.int32,(None, self.config.seq_length))
        mask_placeholder =  tf.placeholder(tf.bool,(None,self.config.seq_length))
        self.input_placeholder = input_placeholder
        self.labels_placeholder = labels_placeholder
        self.mask_placeholder = mask_placeholder
        ### 

    def create_feed_dict(self, inputs_batch, labels_batch=None, mask_batch=None):
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
        feed_dict = {self.input_placeholder: inputs_batch, self.mask_placeholder: mask_batch}

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        ### 
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings. 
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """
        ### YOUR CODE HERE
        embedding_tensor = tf.Variable(tf.cast(self.pretrained_embeddings,tf.float32),dtype=tf.float32) #BK: pretrained_embeddings is has shape (n_tokens, embed_size)
        embeddings = tf.nn.embedding_lookup(embedding_tensor, self.input_placeholder) # tensor of shape (None, seq_length, embed_size)
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. 


        """
        enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.cell_size) #temp: can use different types of rnn
        enc_input = tf.cast(self.add_embedding(),tf.float32) # (None, seq_length, embed_size)
        with tf.variable_scope("encoder"):
            _, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_input, None, initial_state=tf.zeros((tf.shape(enc_input)[0], enc_cell.state_size),tf.float32)) #None represents the argument 'sequence_length', which is a tensor of shape [batch_size], which specifies the length of the sequence for each element of the batch. The fourth arg is initial state.

        dec_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.cell_size) #temp: can use different types of rnn
        dec_cell = tf.nn.rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size)

        #Creates a decoder input, for which we append the zero vector at the beginning of each sequence, which serves as the "GO" token.
        unpacked_enc_input = tf.unstack(enc_input, axis=1)
        unpacked_dec_input = [tf.zeros_like(unpacked_enc_input[0])] + unpacked_enc_input[:-1]
        dec_input = tf.stack(unpacked_dec_input, axis=1)
        
        with tf.variable_scope("decoder"):
            dec_output, _ = tf.nn.dynamic_rnn(dec_cell, dec_input, None, enc_state) 

        embed_pred = dec_output #(None, seq_length, embed_size).
        embed_pred = tf.reshape(embed_pred, (-1, self.config.embed_size)) # (None*seq_length, embed_size)
        # transpose of embedding_tensor of shape (embed_size, n_tokens). This is initialized the same way as embedding_tensor, but they are seperate variables.
        un_embedding_tensor = tf.Variable(tf.cast(tf.transpose(self.pretrained_embeddings),dtype=tf.float32),dtype=tf.float32)
        pred_bias = tf.Variable(tf.zeros((self.config.seq_length, self.config.n_tokens)), dtype=tf.float32)
        pred = tf.matmul(embed_pred, un_embedding_tensor) # (None*seq_length, n_tokens)
        pred = tf.reshape(pred, (-1, self.config.seq_length, self.config.n_tokens)) # (None, seq_length, n_tokens)
        pred = pred + pred_bias

        return pred # This is logits for each token.

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        """
        ### 
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, self.labels_placeholder) # labels = (None, seq_length), and pred = (None, seq_length, n_tokens)
        loss = tf.boolean_mask(loss, self.mask_placeholder)
        loss = tf.reduce_mean(loss)
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

    def preprocess_sequence_data(self, examples):
        return self.pad_sequences(examples, self.config.seq_length)     

    def run_epoch(self, sess, train_examples):
        """Runs an epoch of training.

        """
        n_minibatches, total_loss = 0, 0
        for batches in minibatches(train_examples, self.config.batch_size): #train_examples is a list of (sentence, label, mask) tuples. Each setence/label/mask is itself a list of integers or boolean(in case of mask).
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, *batches) #batches is a list of input_batch, labels_batch, mask_batch, each of which is np.array. That is, [input_batch, labels_batch, mask_batch].
        return total_loss / n_minibatches

    def fit(self, sess, train_examples_raw):
        """Fit model on provided data.

        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            train_examples = self.preprocess_sequence_data(train_examples_raw) #train_examples_raw means unpadded data, which is a list of (sentence, labels) tuples. The length of sentence and labels vary within the list. Sentence and labels are themselves a list of integers representing tokens.
            average_loss = self.run_epoch(sess, train_examples)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch+1, average_loss, duration)
            losses.append(average_loss)
        return losses

    def __init__(self, config, pretrained_embeddings):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()

def generate_sequence_data(n_tokens, train_size, seq_length, n_features=1):
    """Each element is drawn from i.i.d Gaussians.
    Note that train_size, the size of the entire training set, is in general different from and much larger than batch_size, which refers to the size of minibatches.
    """
    return np.random.randint(n_tokens, size=(train_size, seq_length, n_features))

def do_seq2seq_prediction():
    config = Config()
    n_tokens = config.n_tokens
    embed_size = config.embed_size
    pretrained_embeddings = np.random.randn(n_tokens, embed_size) #temp: later this, together with n_tokens and embed_size, has to be replaced with real word2vec.
    #Generate a training data
    np.random.seed(0)
    inputs = np.reshape(generate_sequence_data(n_tokens, 100*config.batch_size, config.seq_length), (100*config.batch_size, config.seq_length))
    labels = inputs
    train_examples_raw = zip(inputs.tolist(),labels.tolist()) # This is a list of (input, label) tuples.
    #Create and train a seq2seq autoencoder
    with tf.Graph().as_default():
        model = Seq2seq_autoencoder(config, pretrained_embeddings)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            model.fit(sess, train_examples_raw)

if __name__ == "__main__":
    do_seq2seq_prediction()