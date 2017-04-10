import random

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import embedding_ops
from tensorflow.models.rnn.translate import data_utils
import paradet_helper
from enum import Enum

class State(Enum):
    TRAIN = 1
    QUALVEC = 2
    TRANSL = 3
    TEST = 4

class QualVecModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               embedding_size,
               buckets,
               size,
               maxout_size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               state = State.TRAIN,
               num_samples=512,
               dtype=tf.float32):
    """Create the model.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, embedding_size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(labels, inputs):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                       num_samples, self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss

    # Create the internal multi-layer cell for our RNN.
    for_cell = tf.nn.rnn_cell.GRUCell(size)
    bac_cell = tf.nn.rnn_cell.GRUCell(size)

    
    if num_layers > 1:
        for_cell = tf.nn.rnn_cell.MultiRNNCell([for_cell] * num_layers)
        bac_cell = tf.nn.rnn_cell.MultiRNNCell([bac_cell] * num_layers)

    def qualvec_f(encoder_inputs, decoder_inputs, do_decode):
        return paradet_helper.embedding_attention_qualvec(
            encoder_inputs,
            decoder_inputs,
            for_cell,
            bac_cell,
            num_encoder_symbols=source_vocab_size,
            num_decoder_symbols=target_vocab_size,
            embedding_size=embedding_size,
            maxout_size=maxout_size,
            output_projection=output_projection,
            feed_previous=do_decode,
            dtype=dtype)
    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    # self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None,None],
    #                                             name="encoder"))
    # Add 1 because of the Go token.
    for i in range(buckets[-1][1] + 1):
        self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
        self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))
    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in range(len(self.decoder_inputs) - 1)]
    # Training outputs and losses.
    if state == State.QUALVEC:
        # Remember self.outputs is a list of buckets and each is a list of output tensors
        self.outputs, self.losses, self.attns = qualvec_helper.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets, lambda x, y: qualvec_f(x, y, True),
            softmax_loss_function=softmax_loss_function)
        # Get the output embedding vector and bias for the translated word
        # targets2 = [(embedding_ops.embedding_lookup(w_t, 
        #             self.decoder_inputs[i + 1]),
        #             tf.gather(output_projection[1], self.decoder_inputs[i + 1]))
        #             for i in range(len(self.decoder_inputs) - 1)]
        targets2 = [embedding_ops.embedding_lookup(w_t, 
                    self.decoder_inputs[i + 1])
                    for i in range(len(self.decoder_inputs) - 1)]
        # Calculate the quality vector
        for b in range(len(buckets)):
            # self.outputs[b] = [
            #     tf.concat(1, [(target * output), tf.reshape(bias, [-1, 1])])
            #     for output, (target, bias) in zip(self.outputs[b], targets2)
            # ]
            self.outputs[b] = [target * output
                              for output, target in zip(self.outputs[b], targets2)
            ]
    elif state == State.TRAIN:
        self.stateIn, self.stateOut, _ = paradet_helper.model_with_buckets(
                                self.encoder_inputs, self.decoder_inputs, targets,
                                self.target_weights, buckets,
                                lambda x, y: qualvec_f(x, y, False),
                                softmax_loss_function=softmax_loss_function)
    elif state == State.TRANSL:
        self.stateIns, self.stateOuts, _ = paradet_helper.model_with_buckets(
                                self.encoder_inputs, self.decoder_inputs, targets,
                                self.target_weights, buckets, lambda x, y: qualvec_f(x, y, True),
                                softmax_loss_function=softmax_loss_function)
        
        # for b in range(len(buckets)):
            
        #     self.outputs[b] = [
        #         tf.matmul(output, output_projection[0])
        #         for output in self.outputs[b]
        #     ]

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if state == State.TRAIN:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      # opt = tf.train.RMSPropOptimizer(self.learning_rate)
      # opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.5)
      # opt = tf.train.AdagradOptimizer(self.learning_rate)
      
      
      # for b in range(len(buckets)):
      #   gradients = tf.gradients(self.losses[b], params)
      #   clipped_gradients, norm = tf.clip_by_global_norm(gradients,
      #                                                    max_gradient_norm)
      #   self.gradient_norms.append(norm)
      #   self.updates.append(opt.apply_gradients(
      #       zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())


  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, state):
    """Run a step of the model feeding the given inputs.
    state:
    0 for training
    1 for quality vector
    2 for translation

    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size+1:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size+1:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))
    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in range(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in range(decoder_size+1):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      # input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    # last_target = self.decoder_inputs[decoder_size].name
    # input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if state == State.TRAIN:
        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
        output_feed = [self.stateIns[bucket_id], self.stateOuts[bucket_id]]  # Loss for this batch.
        # Subtract one because I can remove the last output
        # for l in range(decoder_size-1):  # Output quality vectors.
        #     output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if state == State.TRAIN:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    elif state == State.TEST:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
    else:
        # print(outputs[0])
        # embedding_size = outputs[1].shape[1]
        # for l in range(decoder_size-1):
        #     if target_weights[l][0] == 0:
        #         outputs[l+1] = np.zeros((1, embedding_size)).astype(np.float32)
        return None, outputs[0], outputs[1]  # No gradient norm, loss, outputs.
    
  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in range(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
      
      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input)
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size+1):
        batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(self.batch_size, dtype=np.float32)
        for batch_idx in range(self.batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
            # always set weight of last decoder target to 0. Not just input but target
            # Sometimes this will be the EOS symbol. I am chopping this off.
                target = decoder_inputs[batch_idx][length_idx + 1]
            # Set weight = 0 for last decoder output because that doesn't predict anything
            if length_idx >= decoder_size - 1 or target == data_utils.PAD_ID or \
                target == data_utils.EOS_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

  def get_decode_batch(self, data, bucket_id):
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in range(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
      
      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input)
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size+1):
        batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(self.batch_size, dtype=np.float32)
        for batch_idx in range(self.batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
            # always set weight of last decoder target to 0. Not just input but target
            # Sometimes this will be the EOS symbol. I am chopping this off.
                target = decoder_inputs[batch_idx][length_idx + 1]
            # Set weight = 0 for last decoder output because that doesn't predict anything
            if length_idx >= decoder_size - 1 or target == data_utils.PAD_ID or \
                target == data_utils.EOS_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
