import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear  # pylint: disable=protected-access

def _embed(embedding,
           update_embedding=True):
    def embedFunc(y):
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return embedFunc
    
def maxout(inputs, num_units, axis=None):
    """ maxout activation function

    """
    shape = inputs.get_shape().as_list()
    if axis is None:
        # Assume that channel is the last dimension
        axis = -1
    if not shape[0]:
        shape[0] = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not a multiple of num_units({})'
            .format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    x  = tf.reshape(inputs, shape)
    outputs = tf.reduce_max(x, -1, keep_dims=False)
    return outputs

def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.
    """

    def loop_function(prev, _):
        if output_projection is not None:
            prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
        prev_symbol = math_ops.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev

    return loop_function

def attention_decoder(decoder_inputs,
                    initial_for_state,
                    initial_bac_state,
                    attention_states,
                    for_cell,
                    bac_cell,
                    maxout_size,
                    output_size=None,
                    num_heads=1,
                    loop_function=None,
                    embed_function=None,
                    dtype=None,
                    scope=None,
                    embedding_size=620,
                    initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s" %
                     attention_states.get_shape())
  if output_size is None:
    output_size = for_cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    # This is the number of encoders
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]
    # This is the output dimension of each encoder
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    # hidden is just the attention_states reshaped to 4 dimensions
    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])
    # hidden_features = []
    # v = []
    # Divide by two because attention_vec_size consists of both forward 
    # and backward encoder    
    attention_vec_size = attn_size//2  # Size of query vectors for attention.
    # for a in range(num_heads):
    k = variable_scope.get_variable("AttnW_0",
                                      [1, 1, attn_size, attention_vec_size])
    hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
    # v = variable_scope.get_variable("AttnV_0", [attention_vec_size])

    # Set the end state of the reverse encoder as the initial state of decoder
    # Create a two layer RELU Feedforward network.
    # Uncomment below to not use FW
    # state = initial_bac_state
    # state = [None] * len(initial_state)
    state_size = initial_bac_state[0].get_shape()[1].value

    def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        # ds = []  # Results of attention reads will be stored here.
        # if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        #     query_list = nest.flatten(query)
        #     for q in query_list:  # Check that ndims == 2 if specified.
        #         ndims = q.get_shape().ndims
        #         if ndims:
        #             assert ndims == 2
        #     query = array_ops.concat_v2(query_list, 1)
        # for a in range(num_heads):
        with variable_scope.variable_scope("Attention_0"):
            # y = linear(query, attention_vec_size, True)
            y = array_ops.reshape(query, [-1, 1, 1, attention_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum(tf.multiply(y,hidden_features),
                                [2, 3])
            # s = math_ops.reduce_sum(v * math_ops.tanh(y*hidden_features),
            #                       [2, 3])
            a = nn_ops.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = math_ops.reduce_sum(
                array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
            ds = array_ops.reshape(d, [-1, attn_size])
        return ds, a

    outputs = []
    prev = None
    batch_attn_size = array_ops.pack([batch_size, attn_size])
    # attns = [
    #     array_ops.zeros(
    #         batch_attn_size, dtype=dtype) for _ in range(num_heads)
    # ]
   
    # state = initial_bac_state
    hidState = [None] * len(initial_for_state)
    state = [None] * len(initial_for_state)

    for i in range(len(initial_for_state)):
        with variable_scope.variable_scope("F_init_for_%d" % i):
            hidState[i] = tf.nn.relu(linear(initial_for_state[i], state_size, True, scope="Linear0"), name="relu0")
            state[i] = tf.nn.relu(linear(hidState[i], state_size, True, scope="Linear1"), name="relu1")
            # state[i] = tf.nn.relu(linear(y, state_size, True, scope="Linear1"), name="relu1")
    # state = for_cell.zero_state(batch_size, dtype)
      
    # for a in attns:  # Ensure the second shape of attention vectors is set.
    #     a.set_shape([None, attn_size])
        # For first attention, use the input hidden state from backward decoder
    cell_output = state[0]

    contexts = []
    for_output = []
    collect_attn = []
    with variable_scope.variable_scope("Decoder_For"):
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            # if loop_function is not None and prev is not None:
            #     with variable_scope.variable_scope("loop_function", reuse=True):
            #         inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            # if input_size.value is None:
            #     raise ValueError("Could not infer input size from input: %s" % inp.name)
                # Below should almost always be false   
            
            # x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            context, a = attention(cell_output)
            contexts.append(context)
            collect_attn.append(a)
            
            inp_concat = tf.concat(1,[inp, context])
            cell_output, state = for_cell(inp_concat, state)
            for_output.append(cell_output)
            
            # Run the attention mechanism.
            # with variable_scope.variable_scope("AttnOutputProjection"):
            #     # attns is a list of heads. Here I just have one though
            #     output = linear([cell_output] + attns, maxout_size, True)
                # output is t
                # output = maxout(t_tilda, maxout_size)
            # output = linear(t, output_size, True)
            
            # if loop_function is not None:
            #     prev = output
            # outputs.append(output)

    # state = initial_for_state
    
    # state = initial_for_state
    hidState = [None] * len(initial_bac_state)
    state = [None] * len(initial_bac_state)
    for i in range(len(initial_bac_state)):
        with variable_scope.variable_scope("F_init_bac_%d" % i):
            hidState[i] = tf.nn.relu(linear(initial_bac_state[i], state_size, True, scope="Linear0"), name="relu0")
            state[i] = tf.nn.relu(linear(hidState[i], state_size, True, scope="Linear1"), name="relu1")
            # state[i] = tf.nn.relu(linear(y, state_size, True, scope="Linear1"), name="relu1")
    # state = bac_cell.zero_state(batch_size, dtype)
    bac_output = []
    with variable_scope.variable_scope("Decoder_Back"):
        # for i, (inp, out) in enumerate(zip(reversed(input_attn[2:]), reversed(output_attn[:-2]))):
        for i, (inp, context) in enumerate(zip(reversed(decoder_inputs[2:]), reversed(contexts[:-2]))):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            inp_concat = tf.concat(1,[inp, context])
            cell_output, state = bac_cell(inp_concat, state)
            bac_output.insert(0, cell_output)

    q_vec = []
    with variable_scope.variable_scope("OutputProjection"):
        for i, (for_out, bac_out, context) in enumerate(zip(for_output, bac_output, contexts)):
        # for i, (inp, out) in enumerate(zip(reversed(input_attn[2:]), reversed(output_attn[:-2]))):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            t_tilda = linear(tf.concat(1, [for_out, bac_out, decoder_inputs[i], decoder_inputs[i+2], context]), 2*maxout_size, True, scope="sj")
            # temp2 = linear(tf.concat([decoder_inputs[i], decoder_inputs[i+2]]), 2*maxout_size, True, scope="ey")
            # temp3 = linear(context, 2*maxout_size, True, scope="ctxt")
            # t_tilda = temp + temp2 + temp3
            t_output = maxout(t_tilda, maxout_size)
            output = linear(t_output, embedding_size, True, scope="t_tilda2")  
            q_vec.append(output)
    # q_vec = q_vec[::-1]
    # print('collect_attn', len(collect_attn))
    return q_vec, state, collect_attn
    # , [raw_decoder_inputs[0:4], decoder_inputs[0:4],
             # list(input_attn[0:4]), list(reversed(input_attn[2:6]))]


def embedding_attention_decoder(decoder_inputs,
                                initial_for_state,
                                initial_bac_state,
                                attention_states,
                                for_cell,
                                bac_cell,
                                num_symbols,
                                embedding_size,
                                maxout_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.
    """
    if output_size is None:
        output_size = for_cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])
    with variable_scope.variable_scope(
        scope or "embedding_attention_decoder", dtype=dtype) as scope:

        embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
            embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None
        embed_function = _embed(embedding, False)
    
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
        ]
        return attention_decoder(
            emb_inp,
            initial_for_state,
            initial_bac_state,
            attention_states,
            for_cell,
            bac_cell,
            maxout_size=maxout_size,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            embed_function=embed_function,
            embedding_size=embedding_size,
            initial_state_attention=initial_state_attention)

def embedding_attention_qualvec(
            encoder_inputs,
            decoder_inputs,
            for_cell,
            bac_cell,
            num_encoder_symbols,
            num_decoder_symbols,
            embedding_size,
            maxout_size,
            num_heads=1,
            output_projection=None,
            feed_previous=False,
            dtype=None,
            scope=None,
            initial_state_attention=False):
    """Embedding sequence-to-sequence model with attention.

    """
    if output_projection is None:
        raise ValueError("Must provide output projection")
    with variable_scope.variable_scope(
        scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.

        # encoder_outputs, encoder_state = rnn.rnn(encoder_cell,
        #                                          encoder_inputs,
        #                                          dtype=dtype)

        embedding = variable_scope.get_variable("embedding",
                                            [num_encoder_symbols, embedding_size])

        encoder_inputs = [embedding_ops.embedding_lookup(embedding, i)
                          for i in encoder_inputs]

        encoder_outputs, encoder_state_fw, encoder_state_bw = tf.nn.bidirectional_rnn(
                for_cell, bac_cell, encoder_inputs, dtype=dtype)
        # encoder_outputs = tf.concat(2, outputs)
        # encoder_outputs, encoder_state_fw = tf.nn.rnn(
        #         for_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [
            array_ops.reshape(e, [-1, 1, 2*for_cell.output_size]) 
            for e in encoder_outputs
        ]
        # Dimension 1 are the encoder outputs. Dim2 are the 
        attention_states = tf.concat_v2(top_states, 1)

        # Decoder.
        output_size = None

        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(
                decoder_inputs,
                encoder_state_fw,
                encoder_state_bw,
                attention_states,
                for_cell,
                bac_cell,
                num_decoder_symbols,
                embedding_size,
                maxout_size=maxout_size,
                num_heads=num_heads,
                output_size=output_size,
                output_projection=output_projection,
                feed_previous=feed_previous,
                initial_state_attention=initial_state_attention)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, state = embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state_fw,
                    encoder_state_bw,
                    attention_states,
                    for_cell,
                    bac_cell,
                    num_decoder_symbols,
                    embedding_size,
                    maxout_size=maxout_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False,
                    initial_state_attention=initial_state_attention)
            state_list = [state]
            if nest.is_sequence(state):
                state_list = nest.flatten(state)
            return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
          state = nest.pack_sequence_as(
              structure=encoder_state, flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=target)
      else:
        crossent = softmax_loss_function(target, logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       qualvec,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))
  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  # comment out below to turn off
  attns = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _, attn = qualvec(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]+1])
        outputs.append(bucket_outputs)
        attns.append(attn)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]-1],
                  weights[:bucket[1]-1],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]-1],
                  weights[:bucket[1]-1],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses, attns
