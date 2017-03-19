import tensorflow as tf
import softmax_classifier as sf
import my_embed_seq2seq_autoencoder as seq
from utils.general_utils import minibatches
from data_utils import ModelHelper, load_data, load_embeddings
import argparse

# def do_encoding(sess, helper, config, embeddings):
# 	seq_autoencoder = seq.Seq2seq_autoencoder(helper, config, embeddings)
# 	feed = seq_autoencoder.create_feed_dict(inputs_batch, labels_batch=None, mask_batch=mask_batch)
# 	encoding = seq_autoencoder.add_encoding_op() #(None, state_size)
# 	encoded_input = sess.run(encoding, feed_dict=feed)
# 	return encoded_input #(None, state_size)

def load_labels(data_path):
    labels = []
    with open(data_path) as f:
        for line in f:
            labels.append(int(line.strip())+2)
    return labels

def load_and_preprocess_data(args):
    # Build tok2id and id2tok mapping
    print "Loading training data..."
    helper = ModelHelper.load(args.model_path)

    # Process all the input data
    # Convert all the sentences into sequences of token ids
    train_sents = load_data(args.data_train +"/sents.txt")
    train_labels = load_labels(args.data_train +"/labels.txt")

    dev_sents = load_data(args.data_dev +"/sents.txt")
    dev_labels = load_labels(args.data_dev +"/labels.txt")

    test_sents = load_data(args.data_test +"/sents.txt")
    test_labels = load_labels(args.data_test +"/labels.txt")

    print 'train size:', len(train_sents)
    print 'dev size:', len(dev_sents)
    print 'test size:', len(test_sents)

    return helper, train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels


def compute_encoded_inputs(args):
    raise NotImplementedError("TBD")

def labels_helper(args):
    raise NotImplementedError("TBD")


def do_train_and_test(args):
    # sents data is a list of an integer list representing a sentence.
    # labels data is a list of an integer.
    helper, train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels = load_and_preprocess_data(args) 
    
    # Load pre-trained seq2seq model
    seq2seq_config = seq.build_seq2seq_config(args)
    helper = ModelHelper.load(args.model_path)

    embeddings = load_embeddings(args, helper)
    seq2seq_config.n_tokens = embeddings.shape[0]
    seq2seq_config.embed_size = embeddings.shape[1]

    # Initialize softmax classifier model
    softmax_config = sf.SoftmaxConfig()
    softmax_config.input_size = seq2seq_config.cell_size
    softmax_config.n_classes = 5

    with tf.Graph().as_default():
        print "Building autoencoder model..."
        autoencoder = seq.Seq2seq_autoencoder(helper, seq2seq_config, embeddings)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        train_encs = None
        dev_encs = None
        test_encs = None
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, autoencoder.config.model_output)
            train_encs = autoencoder.encode(sess, train_sents)
            dev_encs = autoencoder.encode(sess, dev_sents)
            test_encs = autoencoder.encode(sess, test_sents)


    print 'Finished encoding'

    with tf.Graph().as_default():
        model = sf.SoftmaxModel(softmax_config)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            model.fit(sess, train_encs, train_labels)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests sentiment analysis.')
    subparsers = parser.add_subparsers()

    # Training
    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', default="sst_data/train", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', default="sst_data/dev", help="Dev data")
    command_parser.add_argument('-dts', '--data-test', default="sst_data/test", help="Test data")
    command_parser.add_argument('-mp', '--model-path', default="model_path", help="Model path") #temp

    command_parser.add_argument('-v', '--vocab', default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.set_defaults(func=do_train_and_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
