import softmax_classifier as sf
import my_embed_seq2seq_autoencoder as seq
from utils.general_utils import minibatches
from data_utils.py import ModelHelper

# def do_encoding(sess, helper, config, embeddings):
# 	seq_autoencoder = seq.Seq2seq_autoencoder(helper, config, embeddings)
# 	feed = seq_autoencoder.create_feed_dict(inputs_batch, labels_batch=None, mask_batch=mask_batch)
# 	encoding = seq_autoencoder.add_encoding_op() #(None, state_size)
# 	encoded_input = sess.run(encoding, feed_dict=feed)
# 	return encoded_input #(None, state_size)

def read_txt_data(data_path):
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(line)
    return data

def load_and_preprocess_data(args):
    # Build tok2id and id2tok mapping
    print "Loading training data..."
    helper = ModelHelper.load(args.model_path)

    # Process all the input data
    # Convert all the sentences into sequences of token ids
    train_sents = read_txt_data(args.data_train +"/sents.txt")
    train_labels = read_txt_data(args.data_train +"/labels.txt")
    train_sents = helper.vectorize(train_sents)
    train_labels = helper.vectorize(train_labels)

    dev_sents = read_txt_data(args.data_dev +"/sents.txt")
    dev_labels = read_txt_data(args.data_dev +"/labels.txt")
    dev_sents = helper.vectorize(dev_sents)
    dev_labels = helper.vectorize(dev_labels)

    test_sents = read_txt_data(args.data_test +"/sents.txt")
    test_labels = read_txt_data(args.data_test +"/labels.txt")
    test_sents = helper.vectorize(test_sents)
    test_labels = helper.vectorize(test_labels)


    print 'train size:', len(train_sents)
    print 'dev size:', len(dev_sents)
    print 'test size:', len(test_sents)

    return helper, train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels


def compute_encoded_inputs(args)
    raise NotImplementedError("TBD")

def labels_helper(args)
    raise NotImplementedError("TBD")


def do_train_and_test(args):
    # sents data is a list of an integer list representing a sentence.
    # labels data is a list of an integer.
    helper, train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels = load_and_preprocess_data(args) 
    
	inputs = compute_encoded_inputs(args) # np.array of shape (None, input_size), dtype = float32.
	labels = labels_helper(args) # np.array of shape (None, n_classes), dtype = int32. One-hot vector in each row.
	softmax_config = sf.SoftmaxConfig()
    softmax_config.input_size = 100 # This has to later match the size of RNN.
    softmax_config.n_classes = 5
    #Create and train a seq2seq autoencoder
    with tf.Graph().as_default():
        print "Building model..."
        model = sf.SoftmaxModel(softmax_config)
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
    command_parser.add_argument('-dts', '--data-test', default="sst_data/test", help="Test data")
    command_parser.add_argument('-mp', '--model-path', default="model_path", help="Model path") #temp
    command_parser.set_defaults(func=do_train_and_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)