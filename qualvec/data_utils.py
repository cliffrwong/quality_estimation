import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  # words = []
  words = sentence.strip().split()
  # for space_separated_fragment in sentence.strip().split():
     # words.extend(_WORD_SPLIT.split(space_separated_fragment))
  #   words.append(space_separated_fragment)
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_str(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub("0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        print(len(vocab_list))

        vocab_list = vocab_list[:max_vocabulary_size]
        print(vocab[vocab_list[-1]])
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]

countUNKS = True

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



def prepare_data(data_dir, source_vocabulary_size, target_vocabulary_size, 
    source_lang, target_lang, tokenizer=None):
    
    train_prefix = "train2."
    dev_prefix = "newstest2013."

    train_path = os.path.join(data_dir, train_prefix)
    dev_path = os.path.join(data_dir, train_prefix)
    
    source_train = train_path + source_lang
    target_train = train_path + target_lang
    
    source_dev = dev_path + source_lang
    target_dev = dev_path + target_lang
    
    for file in [source_train, target_train, source_dev, target_dev]:
        if not gfile.Exists(file):
            raise("training files not available")
    # Create vocabularies of the appropriate sizes.
    source_vocab_path = os.path.join(data_dir, "vocab{0}.{1}".format(source_vocabulary_size, source_lang))
    target_vocab_path = os.path.join(data_dir, "vocab{0}.{1}".format(target_vocabulary_size, target_lang))
    
    if not gfile.Exists(source_vocab_path):
        create_vocabulary(source_vocab_path, source_train, source_vocabulary_size, tokenizer)
    if not gfile.Exists(target_vocab_path):
        create_vocabulary(target_vocab_path, target_train, target_vocabulary_size, tokenizer)
    
    # Create token ids for the training data.
    source_train_ids_path = "{0}ids{1}.{2}".format(train_path, source_vocabulary_size, source_lang)
    target_train_ids_path = "{0}ids{1}.{2}".format(train_path, target_vocabulary_size, target_lang)
    
    if not gfile.Exists(source_train_ids_path):
        data_to_token_ids(source_train, source_train_ids_path, source_vocab_path, tokenizer)
    if not gfile.Exists(target_train_ids_path):
        data_to_token_ids(target_train, target_train_ids_path, target_vocab_path, tokenizer)
    
    # Create token ids for the development data.
    source_dev_ids_path = "{0}ids{1}.{2}".format(dev_path, source_vocabulary_size, source_lang)
    target_dev_ids_path = "{0}ids{1}.{2}".format(dev_path, target_vocabulary_size, target_lang)
    
    if not gfile.Exists(source_dev_ids_path):
        data_to_token_ids(source_dev, source_dev_ids_path, source_vocab_path, tokenizer)
    if not gfile.Exists(target_dev_ids_path):
        data_to_token_ids(target_dev, target_dev_ids_path, target_vocab_path, tokenizer)
    
    return (source_train_ids_path, target_train_ids_path,
          source_dev_ids_path, target_dev_ids_path,
          source_vocab_path, target_vocab_path)


def main():
    dir_path = '/home/cliffrwong/Downloads/'
    vocab_path = dir_path + 'pubmed_vocab_en_40000.txt'
    data_path = dir_path + 'pubmed_en_token.txt'
    create_vocabulary(vocab_path, data_path, 40000)
  

if __name__ == "__main__":
    main()