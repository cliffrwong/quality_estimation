import math
import os
import random
import sys
import time
import logging

import numpy as np
import tensorflow as tf
from random import shuffle
import pickle

from tensorflow.python import debug as tf_debug
import qe_model
from tensorflow.python.training import saver as save_mod

from collections import deque
from qe_model import State

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("output_classes", 5,
                            "Number of output classes.")
tf.app.flags.DEFINE_integer("size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 620, "Word Embedding Dimension.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 5000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("qescore", False,
                            "Set to True for prediction.")
tf.app.flags.DEFINE_boolean("combine_ft", False,
                            "Combine the feature file and target file into 1 for all datasets.")
tf.app.flags.DEFINE_boolean("regression", True,
                            "Set to True for regression training. False for classification.")
tf.app.flags.DEFINE_boolean("split_data", False,
                            "Split data set into training, development, and test set.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
_buckets = [9, 14, 24, 49, 69, 79, 99]

def buckify_data(data):
    data_set = [[] for _ in _buckets]
    for features, label in data:
        for bucket_id, input_size in enumerate(_buckets):
            if len(features) <= input_size:
                data_set[bucket_id].append((features, label))
                break
    return data_set

def create_model(session, state):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = qe_model.QEModel(
                          _buckets,
                          FLAGS.size,
                          FLAGS.embedding_size,
                          FLAGS.num_layers,
                          FLAGS.output_classes,
                          FLAGS.max_gradient_norm,
                          FLAGS.batch_size,
                          FLAGS.learning_rate,
                          FLAGS.learning_rate_decay_factor,
                          FLAGS.regression,
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

# def load_and_enqueue(sess, enqueue_op, coord):
#   with open('dummy_data/features.bin') as feature_file, open('dummy_data/labels.bin') as label_file:
#     while not coord.should_stop():
#       feature_array = np.fromfile(feature_file, np.float32, 128)
#       if feature_array.shape[0] == 0:
#         print('reach end of file, reset using seek(0,0)')
#         feature_file.seek(0,0)
#         label_file.seek(0,0)
#         continue
#       label_value = np.fromfile(label_file, np.float32, 128)

#       sess.run(enqueue_op, feed_dict={feature_input: feature_array,
#                                       label_input: label_value})


def combine_ft_helper(feature_file, target_file, output_file):
    data_set = []
    features = pickle.load(open(FLAGS.data_dir+feature_file, "rb"))
    with open(FLAGS.data_dir+target_file, "r" ) as flabels:
        for feature, label in zip(features, flabels):
            data_set.append((feature, float(label)))
    pickle.dump(data_set, open(FLAGS.data_dir+output_file, "wb"))
    
def combine_ft():
    combine_ft_helper('qualvec_train.txt', 'train.hter', 'train_set.p')
    combine_ft_helper('qualvec_dev.txt', 'dev.hter', 'develop_set.p')
    combine_ft_helper('qualvec_test.txt', 'test.hter', 'test_set.p')
    
def split_data():
    data_set = []
    qualVec = pickle.load( open(FLAGS.data_dir+'qualvec0.txt', "rb" ) )
    with open(FLAGS.data_dir+'ter_class.data', "r" ) as flabels:
        for features, label in zip(qualVec, flabels):
            data_set.append((features, label))
    dataLen = len(data_set)
    indices = list(range(dataLen))
    shuffle(data_set)
    print('length of data ', len(data_set))
    p1 = int(.8*float(dataLen))
    p2 = int(.9*float(dataLen))
    print(dataLen, p1, p2)
    
    train_set = data_set[:p1]
    develop_set = data_set[p1:p2]
    test_set = data_set[p2:]

    pickle.dump( train_set, open( FLAGS.data_dir+"train_set.p", "wb" ) )
    pickle.dump( develop_set, open( FLAGS.data_dir+"develop_set.p", "wb" ) )
    pickle.dump( test_set, open( FLAGS.data_dir+"test_set.p", "wb" ) )

def train():
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, State.TRAIN)
    sess.graph.finalize()
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    # Read data into buckets and compute their sizes.
    
    train_set = buckify_data(pickle.load(open(FLAGS.data_dir+"train_set.p", "rb")))
    dev_set = buckify_data(pickle.load(open(FLAGS.data_dir+"develop_set.p", "rb")))
    
    # test_set = pickle.load( open( "test_set.p", "rb" ) )
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    dev_bucket_sizes = [len(dev_set[b]) for b in range(len(_buckets))]
    dev_total_size = float(sum(dev_bucket_sizes))
    dev_buckets_frac = [i/dev_total_size for i in dev_bucket_sizes]
    
    # This is the training loop.
    step_time, mse_train = 0.0, 0.0
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
      inputs, labels = model.get_batch(train_set, bucket_id)
        
      # score is rmse if regression. loss is mse for regression
      # else it is accuracy for classification.
      _, step_loss, score_train = model.step(
                                    sess, inputs, labels, bucket_id, State.TRAIN,
                                    FLAGS.regression)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      # score_train += score_train / FLAGS.steps_per_checkpoint
      mse_train += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        rmse_train = math.sqrt(mse_train)
        print ("global step %d learning rate %.4f step-time %.2f rmse "
               "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, rmse_train))

        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and rmse_train > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
        previous_losses.append(rmse_train)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "qualScore.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        dev_score = 0.0
        
        for bucket_id in range(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
                print(" eval: empty bucket %d" % (bucket_id))
                continue
            inputs, labels = model.get_batch(dev_set, bucket_id)
            lossList, eval_loss, outputs = model.step(
                                    sess, inputs, labels, bucket_id, State.TEST,
                                    FLAGS.regression)
            for x, y, loss in zip(outputs[:10], labels[:10], lossList[:10]):
                print(x,y,loss)
            # print('output shape', ) 
            dev_score += eval_loss*dev_buckets_frac[bucket_id]
            # dev_loss += eval_loss
            print("  eval: bucket %d score %.2f" % (bucket_id, math.sqrt(eval_loss)))
        # dev_score = math.sqrt(dev_score)
        print(" eval: all bucket rmse: %.2f" % (dev_score))
        sys.stdout.flush()
        # Write to summary writer
        summary_str = tf.Summary(value=[
            tf.Summary.Value(tag="dev. rmse", simple_value=dev_score),
            tf.Summary.Value(tag="train rmse", simple_value=rmse_train)])
        summary_writer.add_summary(summary_str, current_step)
        step_time, mse_train = 0.0, 0.0
        
def qescore():
  with tf.Session() as sess:
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Create model and load parameters.
    # summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    test_set = pickle.load(open(FLAGS.data_dir+"test_set.p", "rb"))
    out_path = os.path.join(FLAGS.data_dir,
                                 "test_prediction.txt")
    # Get token-ids for the input sentences.
    results = []

    model = create_model(sess, State.QESCORE)
    model.batch_size = 1  # We decode one sentence at a time.
    
    # with open(out_path, mode="a") as outfile:
    for feature, label in test_set:
        # Get the bucket id for this sentence
        bucket_id = len(_buckets) - 1
        for i, bucket_size in enumerate(_buckets):
            if len(feature) <= bucket_size:
                bucket_id = i
                break
        else:
            logging.warning("Sentence truncated") 
                                                                                                                                                                                                                                                                                                                                                                                                                         # Get a 1-element batch to feed the sentence to the model.
        inputs, labels = model.get_batch(
            {bucket_id: [(feature, label)]}, bucket_id)
        # Get output logits for the sentence.
        inputs = [np.reshape(x, (-1, FLAGS.embedding_size)) for x in inputs]
        # print('label size', len(labels))
        _, eval_loss, outputs = model.step(
                                    sess, inputs, labels, bucket_id, State.QESCORE,
                                    FLAGS.regression)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        # Write the quality vectors to file
        # outputs2 = [output[0] for output in outputs]
        results.append(outputs)
        # np.savetxt(outfile, outputs)
        # outfile.write("{0}\n".format(outputs))
    # with open(out_path,'wb') as f:
    #     np.savetxt(f, results, fmt='%.5f')
    pickle.dump(results, open(out_path, "wb" ) )


def main(_):
    if FLAGS.split_data:
        split_data()
    elif FLAGS.combine_ft:
        combine_ft()
    elif FLAGS.qescore:
        qescore()
    else:
        train()

if __name__ == "__main__":
  tf.app.run()
