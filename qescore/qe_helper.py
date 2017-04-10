import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

linear = rnn_cell._linear

def qescore_rnn(inputs,
                cell,
                output_classes,
                size,
                regression,
                dtype):
    with variable_scope.variable_scope(
        "qescore_rnn", dtype=dtype) as scope:
        dtype = scope.dtype
        # outputs, state = tf.nn.rnn( 
                # cell, inputs, dtype=dtype)
        cell_bw = tf.nn.rnn_cell.GRUCell(size)
    
        top_outs, state_fw, state_bw = tf.nn.bidirectional_rnn(
                cell, cell_bw, inputs, dtype=dtype)
        
        # if regression:
            # temp = tf.nn.relu(linear(outputs[-1], 100, True, scope="Regression0"))
            # result = linear(temp, 1, True, scope="Regression1")
            
            # result = tf.linear(tf.relu(linear(outputs[-1], 100, True, scope="Regression0")))
            
            # result = tf.scalar_mul(100, tf.sigmoid(linear(outputs[-1], 1, True, scope="Regression0")))
        state = tf.concat(1, [state_fw, state_bw])    
        result = tf.sigmoid(linear(state, 1, True, scope="Regression0"))
        # else:
        #     result = linear(outputs[-1], output_classes, True, scope="Classify0")
        return result, state
      
def calculate_loss(logits,
                  labels,
                  regression,
                  average_across_batch=True,
                  name=None):
  """Weighted cross-entropy loss for logit, batch-collapsed.
  """
  with ops.name_scope(name, "sequence_loss", [logits, labels]):
    # if regression:
        # return squared error
    losses = tf.square(labels-logits)
    loss = tf.sqrt(tf.reduce_mean(losses))
    # else:
    #     # get the cross entropy loss
    #     loss = tf.reduce_mean(
    #             tf.nn.sparse_softmax_cross_entropy_with_logits(
    #             logits=logits, labels=labels))
    return (loss, losses)


def model_with_buckets(inputs,
                        labels,
                        buckets,
                        regression,
                        qescore_f,
                        name=None):
    if len(inputs) < buckets[-1]:
        raise ValueError("Length of inputs (%d) must be at least that of last"
                     "bucket (%d)." % (len(inputs), buckets[-1]))
    losses = []
    lossLists = []
    outputs = []
    with ops.name_scope(name, "model_with_buckets", inputs+[labels]):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
                bucket_outputs, _ = qescore_f(inputs[:bucket])
                outputs.append(bucket_outputs)
                loss, lossList = calculate_loss(outputs[-1], labels, regression)
                losses.append(loss)
                lossLists.append(lossList)
    return outputs, losses, lossList