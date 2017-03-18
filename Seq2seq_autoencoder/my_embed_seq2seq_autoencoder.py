import argparse
from datetime import datetime

import tensorflow as tf
import numpy as np
from seq_model import SeqModel
import time
from utils.general_utils import minibatches
from data_utils import load_and_preprocess_data, load_embeddings
import my_rnn_cell
import my_rnn

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
    batch_size = 20
    n_tokens = 7
    n_epochs = 5000
    lr = 1e-3
    cell_size = None
    cell_type = "rnn" #This can be either "rnn", "gru", or "lstm". 
    cell_init = "random" #This must be either "random" or "identity". Default is "random" and it can be changed in do seq2seq_prediction().
    activation_choice = "relu" #This must be either "relu" or "tanh". 
    feed_decoder = True
    enc_dropout = 0.8
    dec_dropout = 0.8

    clip_gradients = True
    max_grad_norm = 1e3
    sampling = False
    n_sampled = 1000

    def __init__(self, args):
        self.cell_type = args.cell_type
        self.cell_size = args.cell_size
        self.cell_init = args.cell_init
        self.activation_choice = args.activation_choice
        self.clip_gradients = args.clip_gradients
        self.feed_decoder = args.feed_decoder

        if "model_path" in args:
            self.model_path = args.model_path
        else:
            if self.feed_decoder:
                 self.model_path = "results/{}_{}_{}_{}_feed_dec/{:%Y%m%d_%H%M%S}/".format(self.cell_type,
                                                                         self.cell_size,
                                                                         self.cell_init,
                                                                         self.activation_choice,
                                                                         datetime.now())

            else:
                self.model_path = "results/{}_{}_{}_{}/{:%Y%m%d_%H%M%S}/".format(self.cell_type,
                                                                         self.cell_size,
                                                                         self.cell_init,
                                                                         self.activation_choice,
                                                                         datetime.now())

        self.model_output = self.model_path + "model.weights"


class Seq2seq_autoencoder(SeqModel):
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
        _PAD = b"_PAD"
        padding_int = self.helper.tok2id.get(_PAD)

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
        enc_dropout_placeholder = tf.placeholder(tf.float32, ())
        dec_dropout_placeholder = tf.placeholder(tf.float32, ())
        self.input_placeholder = input_placeholder
        self.labels_placeholder = labels_placeholder
        self.mask_placeholder = mask_placeholder
        self.enc_dropout_placeholder = enc_dropout_placeholder
        self.dec_dropout_placeholder = dec_dropout_placeholder

        ### 

    def create_feed_dict(self, inputs_batch, labels_batch=None, mask_batch=None, enc_dropout=1, dec_dropout=1):
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
        feed_dict = {self.input_placeholder: inputs_batch,
                     self.mask_placeholder: mask_batch,
                     self.enc_dropout_placeholder: enc_dropout, 
                     self.dec_dropout_placeholder: dec_dropout}

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
        return embeddings, embedding_tensor

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. 


        """
        enc_dropout_rate = self.enc_dropout_placeholder
        dec_dropout_rate = self.dec_dropout_placeholder

        if self.config.activation_choice == "relu":
            activation = tf.nn.relu
        else:
            activation = tf.nn.tanh
        if self.config.cell_type == "rnn":
            enc_cell = my_rnn_cell.BasicRNNCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.BasicRNNCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size, dec_dropout_rate)
        elif self.config.cell_type == "gru":
            enc_cell = my_rnn_cell.GRUCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.GRUCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size, dec_dropout_rate)
        elif self.config.cell_type == "lstm":
            enc_cell = my_rnn_cell.BasicLSTMCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.BasicLSTMCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size, dec_dropout_rate)                        

        enc_input, embedding_tensor = self.add_embedding() # (None, max_length, embed_size)
        with tf.variable_scope("encoder"):
            """None represents the argument 'sequence_length', which is a tensor of shape [batch_size], which specifies the length of the sequence 
            for each element of the batch. The fourth arg is initial state."""
            _, enc_state = my_rnn.dynamic_rnn(enc_cell, enc_input, None, dtype=tf.float32)

        #Creates a decoder input, for which we append the zero vector at the beginning of each sequence, which serves as the "GO" token.
        if self.config.feed_decoder:
            go_embedding = embedding_tensor[self.helper.tok2id.get("_GO")]
            unpacked_enc_input = tf.unstack(enc_input, axis=1)
            unpacked_dec_input = [tf.zeros_like(unpacked_enc_input[0])+go_embedding] + unpacked_enc_input[:-1]
            dec_input = tf.stack(unpacked_dec_input, axis=1)
        else:
            dec_input = tf.zeros_like(enc_input)

        #Create a drop-out enc_state, which is later fed into the decoder.
        if self.config.cell_type == "lstm":
            c, h = enc_state
            c_dropout = tf.nn.dropout(c, enc_dropout_rate)
            h_dropout = tf.nn.dropout(h, enc_dropout_rate)
            enc_state = my_rnn_cell.LSTMStateTuple(c_dropout, h_dropout)
        else:
            enc_state = tf.nn.dropout(enc_state, enc_dropout_rate)
        
        with tf.variable_scope("decoder"):
            dec_output, _ = my_rnn.dynamic_rnn(dec_cell, dec_input, None, enc_state) 

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

    def add_encoding_op(self):
        
        enc_dropout_rate = self.enc_dropout_placeholder
        dec_dropout_rate = self.dec_dropout_placeholder

        if self.config.activation_choice == "relu":
            activation = tf.nn.relu
        else:
            activation = tf.nn.tanh
        if self.config.cell_type == "rnn":
            enc_cell = my_rnn_cell.BasicRNNCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.BasicRNNCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size, dec_dropout_rate)
        elif self.config.cell_type == "gru":
            enc_cell = my_rnn_cell.GRUCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.GRUCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size, dec_dropout_rate)
        elif self.config.cell_type == "lstm":
            enc_cell = my_rnn_cell.BasicLSTMCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.BasicLSTMCell(self.config.cell_size, self.config.cell_init, activation=activation)
            dec_cell = my_rnn_cell.OutputProjectionWrapper(dec_cell, self.config.embed_size, dec_dropout_rate)                        

        enc_input, embedding_tensor = self.add_embedding() # (None, max_length, embed_size)
        with tf.variable_scope("encoder"):
            """None represents the argument 'sequence_length', which is a tensor of shape [batch_size], which specifies the length of the sequence 
            for each element of the batch. The fourth arg is initial state."""
            _, enc_state = my_rnn.dynamic_rnn(enc_cell, enc_input, None, dtype=tf.float32)

        if self.config.cell_type == "lstm": # In case of LSTM, we average c and h.
            c, h = enc_state
            enc_state = (c + h)/2

        return enc_state 

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
        if self.config.clip_gradients:
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            tvars = tf.trainable_variables()
            original_grads = tf.gradients(loss, tvars) 
            grads, _ = tf.clip_by_global_norm(original_grads, self.config.max_grad_norm)
            train_op = optimizer.apply_gradients(zip(grads, tvars))
        else:
            train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        ###
        return train_op

    def preprocess_sequence_data(self, examples):
        return self.pad_sequences(examples, self.config.max_length)     

    def run_epoch(self, sess, train_examples, dev_set):
        """Runs an epoch of training.

        """
        n_minibatches, total_loss = 0, 0
        for batches in minibatches(train_examples, self.config.batch_size):
        #train_examples is a list of (sentence, label, mask) tuples.
        #Each setence/label/mask is itself a list of integers or boolean(in case of mask).
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, *batches)
            '''
            preds= self.predict_on_batch(sess, *batches)
            for pred in preds:
                print [self.helper.id2tok.get(np.argmax(t)) for t in pred]
            '''
            
            # batches is a list of input_batch, labels_batch, mask_batch.
            # That is, batches = [input_batch, labels_batch, mask_batch].

        train_loss = total_loss / n_minibatches
        dev_loss = self.evaluate(sess, dev_set)

        return train_loss, dev_loss

    def evaluate(self, sess, examples):
        n_minibatches, total_loss = 0, 0
        for batches in minibatches(examples, self.config.batch_size):
            n_minibatches += 1
            total_loss += self.loss_on_batch(sess, *batches)
        return total_loss / n_minibatches

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        """Fit model on provided data.

        """
        # train_examples_raw means unpadded data, which is a list of (sentence, labels) tuples.
        # The length of sentence and labels vary within the list.
        # Sentence and labels are themselves a list of integers representing tokens.
        best_score = float('inf')

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            train_loss, dev_loss = self.run_epoch(sess, train_examples, dev_set)
            duration = time.time() - start_time
            print 'Epoch {:}: train_loss = {:.2f} dev_loss = {:.2f} ({:.3f} sec)'.format(epoch+1, train_loss, dev_loss, duration)
            losses.append(train_loss)

            if dev_loss < best_score:
                best_score = dev_loss
                if saver:
                    print 'Saving model weights...'
                    saver.save(sess, self.config.model_output)

        return losses

    def __init__(self, helper, config, pretrained_embeddings):
        """Initializes the model.

        Args:
            config: A model configuration object of type Config
        """
        self.helper = helper
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.config.max_length = helper.max_length
        self.config.n_tokens = self.pretrained_embeddings.shape[0]
        self.config.embed_size = self.pretrained_embeddings.shape[1]
        self.build()

def generate_sequence_data(n_tokens, train_size, seq_length, n_features=1):
    """Each element is drawn from i.i.d Gaussians.
    Note that train_size, the size of the entire training set, is in general
    different from and much larger than batch_size, which refers to the size of minibatches.
    """
    return np.random.randint(n_tokens, size=(train_size, seq_length, n_features))

def do_train(args):
    config = Config(args)
    print "== Seq2Seq Config =="
    print "  Cell size:", config.cell_size
    print "  Cell type:", config.cell_type
    print "  Cell init:", config.cell_init
    print "  Activation:", config.activation_choice
    print "  Gradient clipping:", config.clip_gradients
    print "  Feed decoder:", config.feed_decoder

    # Load training data
    helper, train, dev = load_and_preprocess_data(args)
    inputs = train
    labels = train
    train_examples_raw = zip(inputs,labels) # This is a list of (input, label) tuples.
    dev_set_raw = zip(dev, dev)

    # Load pretrained embedding matrix
    # Embedding matrix has shape of (n_tokens, embed_size)
    embeddings = load_embeddings(args, helper)
    config.n_tokens = embeddings.shape[0]
    config.embed_size = embeddings.shape[1]
    helper.save(config.model_path)

    #Create and train a seq2seq autoencoder
    with tf.Graph().as_default():
        print "Building model..."
        model = Seq2seq_autoencoder(helper, config, embeddings)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            model.fit(sess, saver, train_examples_raw, dev_set_raw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    # Training
    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', default="data/train.txt", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', default="data/dev.txt", help="Dev data")
    command_parser.add_argument('-v', '--vocab', default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-ct', '--cell-type', choices=["rnn", "gru", "lstm"],
            default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-cs', '--cell-size', default=200, type=int,  help="Size of cell.")
    command_parser.add_argument('-ci', '--cell-init', choices=["random", "identity"],
            default="random", help="Cell initialization.")
    command_parser.add_argument('-ac', '--activation-choice', choices=["tanh", "relu"],
            default="tanh", help="Activation function.")
    command_parser.add_argument('-cg', '--clip-gradients',
            action="store_true",help="Enable gradient clipping.", default=False)
    command_parser.add_argument('-fd', '--feed-decoder', action="store_true", default=False)

    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
