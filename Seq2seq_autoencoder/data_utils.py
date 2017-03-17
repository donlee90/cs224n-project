"""Utilities for downloading data from WMT, tokenizing, vocabularies."""

import os
import re
import pickle

from collections import Counter, OrderedDict
import numpy as np

# Size of embedding vector
EMBED_SIZE = 50

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_NUM = b"_NUM"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    if isinstance(space_separated_fragment, str):
        word = str.encode(space_separated_fragment)
    else:
        word = space_separated_fragment  
    words.extend(re.split(_WORD_SPLIT, word))
  return [w for w in words if w]

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return _NUM
    else: return word.lower()

class ModelHelper(object):
    def __init__(self, tok2id, id2tok, max_length):
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.max_length = max_length

    def vectorize_example(self, sentence):
        sentence_ = [self.tok2id.get(normalize(word), self.tok2id[_UNK])\
                    for word in basic_tokenizer(sentence)]
        return sentence_

    def vectorize(self, data_path):
        with open(data_path) as f:
            return [self.vectorize_example(line) for line in f]

    @classmethod
    def build(cls, data_path):
        # Preprocess data to construct an embedding
        with open(data_path) as f:
            # Initialize dict with special tokens
            tok2id = build_dict(_START_VOCAB)

            # Populate dict with words in the data
            words = [normalize(word) for line in f for word in basic_tokenizer(line)]
            tok2id.update(build_dict(words, offset=len(tok2id)))

            # Build mapping from id to token (for decoder output)
            id2tok = {token_id: word for (word, token_id) in tok2id.items()}

        with open(data_path) as f:
            max_length = max([len(basic_tokenizer(line)) for line in f])

        print "Built dictionary for %d tokens." % (len(tok2id))
        print "Max sequence length: %d" % (max_length)

        return cls(tok2id, id2tok, max_length)

    def save(self, path):
        # Make sure the directory exists.
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id/id2tok map.
        with open(os.path.join(path, "features.pkl"), "w") as f:
            pickle.dump([self.tok2id, self.id2tok, self.max_length], f)

    @classmethod
    def load(cls, path):
        # Make sure the directory exists.
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # Load the tok2id/id2tok map.
        with open(os.path.join(path, "features.pkl")) as f:
            tok2id, id2tok, max_length = pickle.load(f)
        return cls(tok2id, id2tok, max_length)


def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}


def load_and_preprocess_data(args):
    # Build tok2id and id2tok mapping
    print "Loading training data..."
    helper = ModelHelper.build(args.data_train)

    # Process all the input data
    # Convert all the sentences into sequences of token ids
    train_data = helper.vectorize(args.data_train)
    dev_data = helper.vectorize(args.data_dev)
    print 'train size:', len(train_data)
    print 'dev size:', len(dev_data)

    return helper, train_data, dev_data
def load_word_vector_mapping(vocab_fstream, vector_fstream):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.  """
    ret = OrderedDict()
    for vocab, vector in zip(vocab_fstream, vector_fstream):
        vocab = vocab.strip()
        vector = vector.strip()
        ret[vocab] = np.array(list(map(float, vector.split())))

    return ret

def load_embeddings(args, helper):
    embeddings = np.array(np.random.randn(len(helper.tok2id), EMBED_SIZE),
                          dtype=np.float32)
    with open(args.vocab) as vocab, open(args.vectors) as vectors:
        for word, vec in load_word_vector_mapping(vocab, vectors).items():
            word = normalize(word)
            if word in helper.tok2id:
                embeddings[helper.tok2id[word]] = vec
    print "Initialized Embeddings."

    return embeddings
