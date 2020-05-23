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
