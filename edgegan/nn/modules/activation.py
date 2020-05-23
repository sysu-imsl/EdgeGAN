import tensorflow as tf


def activation_fn(input, name='lrelu'):
    assert name in ['relu', 'lrelu', 'tanh', 'sigmoid', None]
    if name == 'relu':
        return tf.nn.relu(input)
    elif name == 'lrelu':
        return tf.maximum(input, 0.2*input)
    elif name == 'tanh':
        return tf.tanh(input)
    elif name == 'sigmoid':
        return tf.sigmoid(input)
    else:
        return input


def miu_relu(x, miu=0.7, name="miu_relu"):
    with tf.variable_scope(name):
        return (x + tf.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


def prelu(x, name="prelu"):
    with tf.variable_scope(name):
        leak = tf.get_variable("param", shape=None, initializer=0.2, regularizer=None,
                               trainable=True, caching_device=None)
        return tf.maximum(leak * x, x)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak * x, x)
