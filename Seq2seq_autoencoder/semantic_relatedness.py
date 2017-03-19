import logistic_regression as lr
import my_embed_seq2seq_autoencoder as seq
from utils.general_utils import minibatches
from data_utils import load_and_preprocess_data, load_embeddings

# def do_encoding(sess, helper, config, embeddings):
# 	seq_autoencoder = seq.Seq2seq_autoencoder(helper, config, embeddings)
# 	feed = seq_autoencoder.create_feed_dict(inputs_batch, labels_batch=None, mask_batch=mask_batch)
# 	encoding = seq_autoencoder.add_encoding_op() #(None, state_size)
# 	encoded_input = sess.run(encoding, feed_dict=feed)
# 	return encoded_input #(None, state_size)

def compute_encoded_inputs(args)
    raise NotImplementedError("TBD")

def labels_helper(args)
    raise NotImplementedError("TBD")


def do_train(args):
	inputs = compute_encoded_inputs(args) # np.array of shape (None, input_size), dtype = float32.
	labels = labels_helper(args) # np.array of shape (None,), dtype = float32.
	logistic_config = lr.LogisticConfig()
    #Create and train a seq2seq autoencoder
    with tf.Graph().as_default():
        print "Building model..."
        model = lr.LogisticModel(logistic_config)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            model.fit(sess, inputs, labels)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests sentiment analysis.')
    subparsers = parser.add_subparsers()

    # Training
    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', default="sst_data/train", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', default="sst_data/dev", help="Dev data")
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