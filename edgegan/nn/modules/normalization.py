import tensorflow as tf


def norm(input, is_train, norm='batch',
         epsilon=1e-5, momentum=0.9, name=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope(name or 'instance_norm'):
            eps = 1e-5
            mean, sigma = tf.nn.moments(input, [1, 2], keep_dims=True)
            normalized = (input - mean) / (tf.sqrt(sigma) + eps)
            out = normalized
    elif norm == 'batch':
        with tf.variable_scope(name or 'batch_norm'):
            out = tf.contrib.layers.batch_norm(input,
                                               decay=momentum, center=True,
                                               updates_collections=None,
                                               epsilon=epsilon,
                                               scale=True, is_training=True)
    else:
        out = input

    return out
