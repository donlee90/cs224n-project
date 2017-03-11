import tensorflow as tf
import numpy as np
from model import Model
import time
from utils.general_utils import minibatches
from data_utils import load_and_preprocess_data, load_embeddings

class Config(object):
# BK: (object) means that Config class inherits from "object" class,
# and in practice I can replace it with ().
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    max_length = 10 # max_length
    embed_size = 50
    batch_size = 50
    n_tokens = 7
    n_epochs = 1000
    lr = 1e-3
    cell_size = 200
    clip_gradients = False
    max_grad_norm = 1e3
    padding_int = 0
    sampling = True
    n_sampled = 1000


class Seq2seq_autoencoder(Model):
# For my own benefit: (Model) means SoftmaxModel class inherits from Model class defined
# in model.py. Therefore, it can use self.build() for e.g.
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
        input_placeholder = tf.placeholder(tf.int32,(None, self.config.max_length))
        labels_placeholder = tf.placeholder(tf.int32,(None, self.config.max_length))
        mask_placeholder =  tf.placeholder(tf.bool,(None,self.config.max_length))
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
        # pretrained_embeddings: tensor of shape (n_tokens, embed_size)
        embedding_tensor = tf.Variable(self.pretrained_embeddings,dtype=tf.float32)

        # embeddings: tensor of shape (None, max_length, embed_size)
        embeddings = tf.nn.embedding_lookup(embedding_tensor, self.input_placeholder)
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. 


        """
        #temp: can use different types of rnn
        enc_cell = tf.nn.rnn_cell.BasicRNNCell(self.config.cell_size)
        enc_input = self.add_embedding() # (None, max_length, embed_size)
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

        embed_pred = dec_output #(None, max_length, embed_size).
        if self.config.sampling:
            pred = embed_pred #(None, max_length, embed_size).
        else:
            # embed_pred: (None*max_length, embed_size)
            embed_pred = tf.reshape(embed_pred, (-1, self.config.embed_size))

            # transpose of embedding_tensor of shape (embed_size, n_tokens).
            # This is initialized the same way as embedding_tensor, but they are seperate variables.
            un_embedding_tensor = tf.Variable(tf.transpose(self.pretrained_embeddings),dtype=tf.float32)
            pred_bias = tf.Variable(tf.zeros((self.config.max_length, self.config.n_tokens)), dtype=tf.float32)
            # pred: (None*max_length, n_tokens)
            pred = tf.matmul(embed_pred, un_embedding_tensor)
            # reshape pred to (None, seq_length, n_tokens)
            pred = tf.reshape(pred, (-1, self.config.max_length, self.config.n_tokens))
            pred = pred + pred_bias

        return pred # This is logits for each token.

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

        """
        ### 
        # labels = (None, max_length), and pred = (None, max_length, n_tokens)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, self.labels_placeholder)
        loss = tf.boolean_mask(loss, self.mask_placeholder)
        loss = tf.reduce_mean(loss)
        ### 
        return loss

    def add_sampled_loss_op(self, pred):
        # pred has shape (None, max_length, embed_size).
        # unstacked_pred is a list of (None, embed_size) tensors.
        unstacked_pred = tf.unstack(pred, axis=1)
        labels = self.labels_placeholder # (None, max_length)
        unstacked_labels = tf.unstack(labels, axis=1) # a list of (None) tensors.
        proj_embedding_tensor = tf.Variable(self.pretrained_embeddings) # This is a tensor of (n_tokens, embed_size), used to project a embedding space vector on vocabulary space. 
        pred_bias = tf.Variable(tf.zeros((self.config.n_tokens,)))
        loss = []
        for each_pred, each_label in zip(unstacked_pred, unstacked_labels):
            each_label = tf.reshape(each_label, (-1,1)) # b/c sampled_softmax_loss demands labels of shape (None, 1). Also, no need to cast into tf.int64, since sampled_softmax_loss does it internally.
            each_loss = tf.nn.sampled_softmax_loss(proj_embedding_tensor, pred_bias, each_pred, each_label, self.config.n_sampled, self.config.n_tokens) # a tensor of shape (None). # Important: this ordering of each_pred, each_label is correct. The website documentation is wrong!
            loss.append(each_loss)
        loss = tf.stack(loss, axis=1) # (None, max_length)
        loss =tf.boolean_mask(loss, self.mask_placeholder) # recall that mask is a (None, max_length)-shaped tensor.
        loss = tf.reduce_mean(loss)
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

        #BK: grads_and_vars is a list of (gradient,variable) pairs. Check if this works.
        grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
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
        return self.pad_sequences(examples, self.config.max_length)     

    def run_epoch(self, sess, train_examples):
        """Runs an epoch of training.

        """
        n_minibatches, total_loss = 0, 0
        for batches in minibatches(train_examples, self.config.batch_size):
        #train_examples is a list of (sentence, label, mask) tuples.
        #Each setence/label/mask is itself a list of integers or boolean(in case of mask).
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, *batches)
            # batches is a list of input_batch, labels_batch, mask_batch.
            # That is, batches = [input_batch, labels_batch, mask_batch].
        return total_loss / n_minibatches

    def fit(self, sess, train_examples_raw):
        """Fit model on provided data.

        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            # train_examples_raw means unpadded data, which is a list of (sentence, labels) tuples.
            # The length of sentence and labels vary within the list.
            # Sentence and labels are themselves a list of integers representing tokens.
            train_examples = self.preprocess_sequence_data(train_examples_raw)
            average_loss = self.run_epoch(sess, train_examples)
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch+1, average_loss, duration)
            losses.append(average_loss)
        return losses

    def __init__(self, helper, config, pretrained_embeddings):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.max_length = helper.max_length
        Config.max_length = self.max_length
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

    # Load training data
    helper, data = load_and_preprocess_data('data/data.txt')
    inputs = data
    labels = inputs
    train_examples_raw = zip(inputs,labels) # This is a list of (input, label) tuples.

    # Load pretrained embedding matrix
    # Embedding matrix has shape of (n_tokens, embed_size)
    embeddings = load_embeddings('data/vocab.txt', 'data/wordVectors.txt', helper)
    config.n_tokens = embeddings.shape[0]
    config.embed_size = embeddings.shape[1]

    #Create and train a seq2seq autoencoder
    with tf.Graph().as_default():
        model = Seq2seq_autoencoder(helper, config, embeddings)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            model.fit(sess, train_examples_raw)

if __name__ == "__main__":
    do_seq2seq_prediction()
