from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
import scipy.spatial.distance
import tensorflow as tf
import pickle

from tensorflow.python import debug as tf_debug
import data_utils
import paradet_model
from tensorflow.python.training import saver as save_mod
from tensorflow.python.ops import embedding_ops

from collections import deque
from paradet_model import State


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 850, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 620, "Word Embedding Dimension.")
tf.app.flags.DEFINE_integer("maxout_size", 500, "Size of maxout layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("source_vocab_size", 40000, "Source vocabulary size.")
tf.app.flags.DEFINE_integer("target_vocab_size", 40000, "Target vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_string("source_lang", "en", "Language.")
tf.app.flags.DEFINE_string("target_lang", "es", "Language.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("qualvec", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("translate", False,
                            "Set to True for trying to translate decoder input.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
FLAGS = tf.app.flags.FLAGS

# if FLAGS.qualvec or FLAGS.translate:
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (60, 70), (70, 80), (90, 100)]
# else:
# _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
_buckets = []

def read_data_decode(source_path, target_path, source_vocab, target_vocab, max_size=None):
    data_set = deque()
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            for source, target in zip(source_file, target_file):
                source_ids = data_utils.sentence_to_token_ids(source, source_vocab)
                target_ids = data_utils.sentence_to_token_ids(target, target_vocab)
                target_ids.append(data_utils.EOS_ID)
                data_set.append([source_ids, target_ids])
    return data_set


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set

def create_model(session, state):
    global _buckets
    # if FLAGS.qualvec or FLAGS.translate:
    #     _buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (60, 70), (70, 80), (90, 100)]
    # else:
    _buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = paradet_model.QualVecModel(
        FLAGS.source_vocab_size,
        FLAGS.target_vocab_size,
        FLAGS.embedding_size,
        _buckets,
        FLAGS.size,
        FLAGS.maxout_size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        state=state,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and save_mod.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train():
  print("Preparing data in %s" % FLAGS.data_dir)
  source_train, target_train, source_dev, target_dev, _, _ = data_utils.prepare_data(
      FLAGS.data_dir, FLAGS.source_vocab_size, FLAGS.target_vocab_size,
      FLAGS.source_lang, FLAGS.target_lang)
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, State.TRAIN)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)

    source_vocab, target_vocab, _ = getVocab()
    # Set max_size to limit training

    train_set = read_data(source_train, target_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]
    
    dev_set = read_data(source_dev, target_dev)
    dev_bucket_sizes = [len(dev_set[b]) for b in range(len(_buckets))]
    dev_total_size = float(sum(dev_bucket_sizes))
    dev_buckets_frac = [i/dev_total_size for i in dev_bucket_sizes]
    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in range(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, State.TRAIN)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))

        # Decrease learning rate if no improvement was seen over last 2 times.
        if len(previous_losses) > 1 and loss > max(previous_losses[-2:]):
            sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        dev_loss = 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in range(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, State.TEST)
          dev_loss += eval_loss*dev_buckets_frac[bucket_id]
          eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
              "inf")
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          
        # Write to summary writer
        dev_ppx = math.exp(float(dev_loss)) if dev_loss < 10 else math.exp(10)
        print(" eval: all bucket perplexity %.2f" % (dev_ppx))
        sys.stdout.flush()

        summary_str = tf.Summary(value=[
          tf.Summary.Value(tag="dev. perplexity", simple_value=dev_ppx),
          tf.Summary.Value(tag="train perplexity", simple_value=perplexity)])
        summary_writer.add_summary(summary_str, current_step)

def getVocab():
    source_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab{0}.{1}".format(
                                    FLAGS.source_vocab_size, FLAGS.source_lang))
    target_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab{0}.{1}".format(
                                    FLAGS.target_vocab_size, FLAGS.target_lang))
    source_vocab, rev_source_vocab = data_utils.initialize_vocabulary(source_vocab_path)
    target_vocab, rev_target_vocab = data_utils.initialize_vocabulary(target_vocab_path)
    return source_vocab, target_vocab, rev_target_vocab

def qualvec():
  with tf.Session() as sess:
    # uncomment for debugging purpose
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Create model and load parameters.
    model = create_model(sess, State.QUALVEC)
    model.batch_size = 1  # We decode one sentence at a time.
    
    # Load vocabularies.
    source_vocab, target_vocab, _ = getVocab()
    
    for set_name in ['train', 'dev', 'test']:
        source_file = os.path.join(FLAGS.data_dir, "{0}2.src".format(set_name))
        target_file = os.path.join(FLAGS.data_dir, "{0}2.mt".format(set_name))
        # output directory to write quality vectors
        out_path = os.path.join(FLAGS.data_dir,
                                     "qualvec_{0}.txt".format(set_name))

        # source_file = os.path.join(FLAGS.data_dir, "eng4.data")
        # target_file = os.path.join(FLAGS.data_dir, "hyp4.data")
        
        # Get token-ids for the input sentences.
        sentences = read_data_decode(source_file, target_file, 
                                      source_vocab, target_vocab)
        results = []

        count = 0
        # with open(out_path, mode="a") as outfile:
        while sentences:
            source_ids, target_ids = sentences.popleft()
            # Get the bucket id for this sentence
            bucket_id = len(_buckets) - 1
            for i, (source_size, target_size) in enumerate(_buckets):
              if len(source_ids) < source_size and len(target_ids) < target_size:
                bucket_id = i
                break
            else:
                logging.warning("Sentence truncated") 

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_decode_batch(
                {bucket_id: [(source_ids, target_ids)]}, bucket_id)
            # Get output logits for the sentence.
            _, _, outputs = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, State.QUALVEC)
                    
            # for x in outputs:
            #     print(x[::-1])
                # print(np.sum(x[:-1]))
                # print(np.sum(x[:-1]))
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            # Write the quality vectors to file
            # outputs2 = [output[0] for output in outputs]
            results.append(outputs)
            # print(outputs)
            # temp = [np.sum(item) for item in outputs]
            # print(temp)
            # temp = [item for item in temp if item != 0]
            # print(sum(temp) / float(len(temp)))
            # temp = [np.sum(item) for item in outputs]
            # print(temp)

            # print(np.mean([item for item in temp if item != 0.0]))
            
            # print([np.sum(np.sqrt(((item) ** 2))) for item in outputs])
            # if count == 20:
            # sys.exit()
            # count += 1
            # print(count)
            # sys.exit()
            # np.savetxt(outfile, outputs)
            # outfile.write("{0}\n".format(outputs))
        # with open(out_path,'wb') as f:
        #     np.savetxt(f, results, fmt='%.5f')
        pickle.dump(results, open(out_path, "wb" ) )

def translate():
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    
        model = create_model(sess, State.TRANSL)
        model.batch_size = 1  # We decode one sentence at a time.
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        
        # Load vocabularies.
        source_vocab, target_vocab, rev_target_vocab = getVocab()
    
        # test_set = read_data(en_train, fr_train, FLAGS.max_train_data_size)
        
        # output directory to write quality vectors
        
        set_name = "train"
        source_file = os.path.join(FLAGS.data_dir, "{0}2.src".format(set_name))
        target_file = os.path.join(FLAGS.data_dir, "{0}2.mt".format(set_name))
        
        # Get token-ids for the input sentences.
        sentences = read_data_decode(source_file, target_file, 
                                      source_vocab, target_vocab)
        results = []
        count = 0
        # with open(out_path, mode="a") as outfile:
        while sentences:
            source_ids, target_ids = sentences.popleft()
            # Get the bucket id for this sentence
            bucket_id = len(_buckets) - 1
            for i, (source_size, target_size) in enumerate(_buckets):
              if len(source_ids) < source_size and len(target_ids) < target_size:
                bucket_id = i
                break
            else:
                logging.warning("Sentence truncated, src len:{0}, tgt len:{1}".format(
                    len(source_ids), len(target_ids)))

            np.set_printoptions(threshold=np.nan)
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_decode_batch(
                {bucket_id: [(source_ids, target_ids)]}, bucket_id)
            # Get output logits for the sentence.
            _, stateIn, stateOut  = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, State.TRANSL)
            # outputs = [int(np.argmax(logit, axis=1)) for logit in outputs if np.any(logit)]
            # print('stateIn', stateIn)
            # print('stateOut', stateOut)
            print(scipy.spatial.distance.cosine(stateIn, stateOut))
            # print(np.dot(stateIn, np.transpose(stateOut)))
            # outputs = [w_t[x] for x in target_ids]
            
            # np.save("bottom", outputs)
            
            # print(target_ids)
            # print('source')
            # print(" ".join([tf.compat.as_str(rev_target_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            count += 1
            if count == 2:
                sys.exit()

            # results.append(outputs)
            # print(outputs)
            # for item in outputs:
            #     print(np.average(item))
            # for item in outputs:
            #     print(np.sqrt(((item) ** 2).mean()))
            
def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = paradet_model.QualVecModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, State.TRAIN, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in range(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.qualvec:
        qualvec()
    elif FLAGS.translate:
        translate()
    else:
        train()

if __name__ == "__main__":
  tf.app.run()
