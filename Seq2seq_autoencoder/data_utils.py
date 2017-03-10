"""Utilities for downloading data from WMT, tokenizing, vocabularies."""

import os
import re

from tensorflow.python.platform import gfile
from collections import Counter, OrderedDict
import numpy as np

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_NUM = b"_NUM"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK, _NUM]

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


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print('>> Full Vocabulary Size :',len(vocab_list))
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

######################################################################

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

class ModelHelper(object):
    def __init__(self, tok2id, id2tok, max_length):
        self.tok2id = tok2id
        self.id2tok = id2tok
        self.max_length = max_length

    def vectorize_example(self, sentence):
        sentence_ = [self.tok2id.get(normalize(word))\
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


def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}


def load_and_preprocess_data(data_path):
    # Build tok2id and id2tok mapping
    print "Loading training data..."
    helper = ModelHelper.build(data_path)

    # Process all the input data
    # Convert all the sentences into sequences of token ids
    data = helper.vectorize(data_path)

    return helper, data

def load_word_vector_mapping(vocab_fstream, vector_fstream):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    """
    ret = OrderedDict()
    for vocab, vector in zip(vocab_fstream, vector_fstream):
        print vocab, vector
        vocab = vocab.strip()
        vector = vector.strip()
        ret[vocab] = np.array(list(map(float, vector.split())))

    return ret

EMBED_SIZE = 10
def load_embeddings(vocab_path, vectors_path, helper):
    print "Loading training data..."
    embeddings = np.array(np.random.randn(len(helper.tok2id), EMBED_SIZE),
                          dtype=np.float32)
    with open(vocab_path) as vocab, open(vectors_path) as vectors:
        for word, vec in load_word_vector_mapping(vocab, vectors).items():
            word = normalize(word)
            if word in helper.tok2id:
                embeddings[helper.tok2id[word]] = vec
    print "Initialized Embeddings."

    return embeddings
