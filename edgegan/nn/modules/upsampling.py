import tensorflow as tf


def upsample(input, data_format):
    assert data_format == 'NCHW'
    output = tf.concat([input, input, input, input], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def upsample2(input, data_format):
    assert data_format == 'NHWC'
    output = tf.transpose(input, [0, 3, 1, 2])
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    return output
