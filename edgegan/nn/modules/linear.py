import tensorflow as tf


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
