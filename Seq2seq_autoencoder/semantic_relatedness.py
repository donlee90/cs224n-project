import argparse
import logistic_regression as lr
import my_embed_seq2seq_autoencoder as seq
from utils.general_utils import minibatches
from data_utils import load_and_preprocess_data, load_embeddings, load_data, ModelHelper
import tensorflow as tf
import numpy as np

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
            labels.append((float(line.strip())-1)/4.0)
    return labels

def load_and_preprocess_data():
    trainA = load_data('data/sick/train/sentsA.txt')
    trainB = load_data('data/sick/train/sentsB.txt')
    train_labels = load_labels('data/sick/train/labels.txt')
    train = (trainA, trainB, train_labels)

    devA = load_data('data/sick/dev/sentsA.txt')
    devB = load_data('data/sick/dev/sentsB.txt')
    dev_labels = load_labels('data/sick/dev/labels.txt')
    dev = (devA, devB, dev_labels)


    testA = load_data('data/sick/test/sentsA.txt')
    testB = load_data('data/sick/test/sentsB.txt')
    test_labels = load_labels('data/sick/test/labels.txt')
    test = (testA, testB, test_labels)


    return train, dev, test

def fuse_encodings(encodingsA, encodingsB):
    assert len(encodingsA) == len(encodingsB), 'Number of encodings does not match'

    fused_encs = []
    for encA, encB in zip(encodingsA, encodingsB):
        fused = np.hstack((encA * encB, np.absolute(encA-encB)))
        fused_encs.append(fused)

    return fused_encs

def do_train(args):

    train, dev, test = load_and_preprocess_data()
    trainA, trainB, train_labels = train
    devA, devB, dev_labels = dev
    testA, testB, test_labels = test

    # Load pre-trained seq2seq model
    seq2seq_config = seq.build_seq2seq_config(args)
    helper = ModelHelper.load(args.model_path)

    embeddings = load_embeddings(args, helper)
    seq2seq_config.n_tokens = embeddings.shape[0]
    seq2seq_config.embed_size = embeddings.shape[1]

    # Initialize regression model
    reg_config = lr.LogisticConfig()
    reg_config.input_size = seq2seq_config.cell_size * 2

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
            trainA_encs = autoencoder.encode(sess, trainA)
            trainB_encs = autoencoder.encode(sess, trainB)
            train_encs = fuse_encodings(trainA_encs, trainB_encs)

            devA_encs = autoencoder.encode(sess, devA)
            devB_encs = autoencoder.encode(sess, devB)
            dev_encs = fuse_encodings(devA_encs, devB_encs)

            testA_encs = autoencoder.encode(sess, testA)
            testB_encs = autoencoder.encode(sess, testB)
            test_encs = fuse_encodings(testA_encs, testB_encs)

    print 'Finished encoding'
    train = [train_encs, train_labels]
    dev = [dev_encs, dev_labels]
    test = [test_encs, test_labels]

    #Create and train a seq2seq autoencoder
    with tf.Graph().as_default():
        print "Building model..."
        model = lr.LogisticModel(reg_config)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            _, test_loss = model.fit(sess, train, dev, test)

    result_path = "sr_results/%s.txt" % args.model_path.split('/')[1]
    with open(result_path, 'w') as result_file:
        result_file.write("%f"%test_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests sentiment analysis.')
    subparsers = parser.add_subparsers()

    # Training
    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-mp', '--model-path', default="model_path", help="Model path") #temp

    command_parser.add_argument('-v', '--vocab', default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
