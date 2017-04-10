import random
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import embedding_ops
from tensorflow.models.rnn.translate import data_utils
from enum import Enum

import qe_helper

class State(Enum):
    TRAIN = 1
    QESCORE = 2
    TEST = 3

class QEModel(object):
  """

  """

  def __init__( self,
                buckets,
                size,
                embedding_size,
                num_layers,
                output_classes,
                max_gradient_norm,
                batch_size,
                learning_rate,
                learning_rate_decay_factor,
                regression,
                state = State.TRAIN,
                num_samples=512,
                dtype=tf.float32):
    """Create the model.
    """
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # Create the internal multi-layer cell for our RNN.
    cell = tf.nn.rnn_cell.GRUCell(size)
    
    
    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([for_cell] * num_layers)
    
    # Feeds for inputs.
    self.inputs = []
    self.output = []
    keep_prob = tf.placeholder(tf.float32)
    for i in range(buckets[-1]):  # Last bucket is the biggest one.
        self.inputs.append(tf.placeholder(tf.float32, shape=[None, embedding_size],
                                                name="input{0}".format(i)))
    if regression:
        self.labels = tf.placeholder(tf.float32, shape=[None],
                                                name="label")
    else:
        self.labels = tf.placeholder(tf.int32, shape=[None],
                                                name="label")
    def qescore_f(inputs):
        return qe_helper.qescore_rnn(inputs,
                                    cell,
                                    output_classes,
                                    size,
                                    regression,
                                    dtype=dtype)

    # Training outputs and losses.
    # if state == State.TEST:
    self.outputs, self.losses, self.lossList = qe_helper.model_with_buckets(
        self.inputs, self.labels, buckets, regression,
        lambda x: qescore_f(x))
    
    

        # If we use output projection, we need to project outputs for decoding.
    # elif state == State.TRAIN:
    #     self.outputs, self.losses = qe_helper.model_with_buckets(
    #         self.inputs, self.labels, buckets, regression,
    #         lambda x: qescore_f(x))

    # Accuracy Operation

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if state == State.TRAIN:
      self.gradient_norms = []
      self.updates = []
      self.scores = []
      # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      # opt = tf.train.RMSPropOptimizer(self.learning_rate)
      opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
      # opt = tf.train.AdagradOptimizer(self.learning_rate)
      
      
      for b in range(len(buckets)):

        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))
        self.scores.append(tf.sqrt(self.losses[b]))
    
    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, inputs, labels, bucket_id, state, regression):
    # Check if the sizes match.
    input_size = self.buckets[bucket_id]
    if len(inputs) != input_size:
        raise ValueError("Input length must be equal to the one in bucket,"
                       " %d != %d." % (len(inputs), input_size))

    #print('output shape', len(self.outputs), self.outputs[0].get_shape())
    
    input_feed = {}
    for l in range(input_size):
        input_feed[self.inputs[l].name] = inputs[l]
    input_feed[self.labels.name] = labels
    if state == State.TRAIN:
        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                        self.gradient_norms[bucket_id],  # Gradient norm.
                        self.losses[bucket_id],  # Loss for this batch.
                        self.scores[bucket_id]] # mean accuracy for this batch or mse
    # elif state == State.TEST:
    else:
        output_feed = [self.losses[bucket_id],
                        self.outputs[bucket_id]]  # Loss for this batch.
        for i in range(64):
            output_feed.append(self.lossList[bucket_id][i])

    outputs = session.run(output_feed, input_feed)
    if state == State.TRAIN:
        return outputs[1], outputs[2], outputs[3]  # Gradient norm, loss, no outputs.
    else:
    # elif state == State.TEST:
        return outputs[2:], outputs[0], outputs[1]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    input_size = self.buckets[bucket_id]
    inputs, labels = [], []
    # Get a random batch of inputs from data,
    for _ in range(self.batch_size):
        features, label = random.choice(data[bucket_id])
        # print([np.average(x) for x in features])
        inputs.append(list(reversed(features)))
        labels.append(label)
    batch_inputs = []
    
    for length_idx in range(input_size):
        batch_inputs.append(
            np.squeeze(np.array([inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.float32)))
    
    return batch_inputs, labels
