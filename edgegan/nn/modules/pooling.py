import tensorflow as tf


def mean_pool(input, data_format):
    assert data_format == 'NCHW'
    output = tf.add_n(
        [input[:, :, ::2, ::2], input[:, :, 1::2, ::2], input[:, :, ::2, 1::2], input[:, :, 1::2, 1::2]]) / 4.
    return output
