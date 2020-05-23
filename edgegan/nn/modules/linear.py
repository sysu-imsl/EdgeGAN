import tensorflow as tf
import tensorflow.contrib.layers as ly
from tensorflow.python.ops import init_ops

from .activation import activation_fn as _activation
from .normalization import norm as _norm
from .normalization import spectral_normed_weight


def linear(input_, output_size, with_w=False, reuse=False, name=None):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name or "linear", reuse=reuse):
        try:
            matrix = tf.get_variable(
                "Matrix", [shape[1], output_size],
                tf.float32,
                tf.random_normal_initializer(stddev=0.02))
        except ValueError as err:
            msg = "NOTE: Usually, this is due to an issue with the image \
dimensions.  Did you correctly set '--crop' or '--input_height' or \
'--output_height'?"
            err.args = err.args + (msg, )
            raise
        bias = tf.get_variable(
            "bias", [output_size],
            initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def fully_connected(inputs, num_outputs, sn, activation_fn=None,
                    normalizer_fn=None, normalizer_params=None,
                    weights_initializer=ly.xavier_initializer(),
                    weight_decay_rate=1e-6,
                    biases_initializer=init_ops.zeros_initializer(),
                    biases_regularizer=None,
                    reuse=None, scope=None, SPECTRAL_NORM_UPDATE_OPS='spectral_norm_update_ops'):
    # TODO move regularizer definitions to model
    weights_regularizer = ly.l2_regularizer(weight_decay_rate)

    input_dim = inputs.get_shape().as_list()[1]

    with tf.variable_scope(scope, 'fully_connected', [inputs], reuse=reuse) as sc:
        inputs = tf.convert_to_tensor(inputs)

        weights = tf.get_variable(name="weights", shape=(input_dim, num_outputs),
                                  initializer=weights_initializer, regularizer=weights_regularizer,
                                  trainable=True, dtype=inputs.dtype.base_dtype)

        # Spectral Normalization
        if sn:
            weights = spectral_normed_weight(
                weights, num_iters=1, update_collection=(SPECTRAL_NORM_UPDATE_OPS))

        linear_out = tf.matmul(inputs, weights)

        if biases_initializer is not None:
            biases = tf.get_variable(name="biases", shape=(num_outputs,),
                                     initializer=biases_initializer, regularizer=biases_regularizer,
                                     trainable=True, dtype=inputs.dtype.base_dtype)

        linear_out = tf.nn.bias_add(linear_out, biases)

        # Apply normalizer function / layer.
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            linear_out = normalizer_fn(
                linear_out, activation_fn=None, **normalizer_params)

        if activation_fn is not None:
            linear_out = activation_fn(linear_out)

    return linear_out


def mlp(input, out_dim, name, is_train, reuse, norm=None, activation=None,
        dtype=tf.float32, bias=True):
    with tf.variable_scope(name, reuse=reuse):
        _, n = input.get_shape()
        w = tf.get_variable('w', [n, out_dim], dtype,
                            tf.random_normal_initializer(0.0, 0.02))
        out = tf.matmul(input, w)
        if bias:
            b = tf.get_variable('b', [out_dim],
                                initializer=tf.constant_initializer(0.0))
            out = out + b
        out = _activation(out, activation)
        out = _norm(out, is_train, norm)
        return out
