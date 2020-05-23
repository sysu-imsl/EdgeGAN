import tensorflow as tf


def conv2d(input, output_dim, filter_size=5, stride=2, reuse=False,
            pad='SAME', bias=True, name=None):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[-1],
                    output_dim]

    with tf.variable_scope(name or 'conv2d', reuse=reuse):
        w = tf.get_variable('w', filter_shape,
                            initializer=tf.truncated_normal_initializer(
                                stddev=0.02))
        if pad == 'REFLECT':
            p = (filter_size - 1) // 2
            x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
        else:
            assert pad in ['SAME', 'VALID']
            conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)

        if bias:
            b = tf.get_variable('b', [output_dim],
                                initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv


def deconv2d(input_, output_shape, with_w=False,
              filter_size=5, stride=2, reuse=False, name=None):
    with tf.variable_scope(name or 'deconv2d', reuse=reuse):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [filter_size, filter_size, output_shape[-1],
                        input_.get_shape()[-1]]

        w = tf.get_variable('w', filter_shape,
                            initializer=tf.random_normal_initializer(
                                stddev=0.02))
        deconv = tf.nn.conv2d_transpose(
            input_, w, output_shape=output_shape, strides=stride_shape)
        b = tf.get_variable('b', [output_shape[-1]],
                            initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        if with_w:
            return deconv, w, b
        else:
            return deconv
