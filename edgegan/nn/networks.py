# -*- coding:utf8 -*-
# defination of network

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
import tensorflow.contrib.layers as ly
import functools
from edgegan.nn import activation_fn as _activation
from edgegan.nn import norm as _norm
from edgegan.nn import conv2d as _conv2d
from edgegan.nn import deconv2d as _deconv2d
from edgegan.nn import linear as _linear
from edgegan.nn import spectral_normed_weight
from edgegan.nn import lrelu

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
    def concat(tensors, axis, *args, **kwargs):
        return tf.concat(tensors, axis, *args, **kwargs)


def conv_block(input, num_filters, name, k_size, stride, is_train, reuse, norm,
               activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = _conv2d(input, num_filters, k_size, stride, reuse, pad, bias)
        out = _norm(out, is_train, norm)
        out = _activation(out, activation)
        return out


def residual(input, num_filters, name, is_train, reuse, norm, pad='REFLECT',
             bias=False):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = _conv2d(input, num_filters, 3, 1, reuse, pad, bias)
            out = _norm(out, is_train, norm)
            out = tf.nn.relu(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = _conv2d(out, num_filters, 3, 1, reuse, pad, bias)
            out = _norm(out, is_train, norm)

        with tf.variable_scope('shortcut', reuse=reuse):
            shortcut = _conv2d(input, num_filters, 1, 1, reuse, pad, bias)

        return tf.nn.relu(shortcut + out)


def residual2(input, num_filters, name, k_size, stride, is_train, reuse, norm,
              activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = _conv2d(input, num_filters, k_size, stride, reuse, pad, bias)
            out = _norm(out, is_train, norm)
            out = _activation(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = _conv2d(out, num_filters, k_size, stride, reuse, pad, bias)
            out = _norm(out, is_train, norm)

        with tf.variable_scope('shortcut', reuse=reuse):
            shortcut = _conv2d(input, num_filters, 1, 1, reuse, pad, bias)

        return _activation(shortcut + out)


def deresidual2(input, num_filters, name, k_size, stride, is_train, reuse,
                norm, activation, with_w=False):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = _deconv2d(input, num_filters, with_w, k_size, stride, reuse)
            out = _norm(out, is_train, norm)
            out = _activation(out, activation)

        with tf.variable_scope('res2', reuse=reuse):
            out = _deconv2d(out, num_filters, with_w, k_size, stride, reuse)
            out = _norm(out, is_train, norm)

        with tf.variable_scope('shortcut', reuse=reuse):
            shortcut = _deconv2d(input, num_filters, with_w, 1, 1, reuse)

        return _activation(shortcut + out, activation)


def deconv_block(input, output_shape, name, k_size, stride, is_train, reuse,
                 norm, activation, with_w=False):
    with tf.variable_scope(name, reuse=reuse):
        out = _deconv2d(input, output_shape, with_w, k_size, stride, reuse)
        out = _norm(out, is_train, norm)
        out = _activation(out, activation)
        return out


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


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])


def mean_pool(input, data_format):
    assert data_format == 'NCHW'
    output = tf.add_n(
        [input[:, :, ::2, ::2], input[:, :, 1::2, ::2], input[:, :, ::2, 1::2], input[:, :, 1::2, 1::2]]) / 4.
    return output


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


def mru_conv_block_v3(inp, ht, filter_depth, sn,
                      stride, dilate=1,
                      activation_fn=tf.nn.relu,
                      normalizer_fn=None,
                      normalizer_params=None,
                      weights_initializer=ly.xavier_initializer_conv2d(),
                      biases_initializer_mask=tf.constant_initializer(
                          value=0.5),
                      biases_initializer_h=tf.constant_initializer(value=-1),
                      data_format='NCHW',
                      weight_decay_rate=1e-8,
                      norm_mask=False,
                      norm_input=True,
                      deconv=False):

    def norm_activ(tensor_in):
        if normalizer_fn is not None:
            _normalizer_params = normalizer_params or {}
            tensor_normed = normalizer_fn(tensor_in, **_normalizer_params)
        else:
            tensor_normed = tf.identity(tensor_in)
        if activation_fn is not None:
            tensor_normed = activation_fn(tensor_normed)

        return tensor_normed

    channel_index = 1 if data_format == 'NCHW' else 3
    reduce_dim = [2, 3] if data_format == 'NCHW' else [1, 2]
    hidden_depth = ht.get_shape().as_list()[channel_index]
    regularizer = ly.l2_regularizer(
        weight_decay_rate) if weight_decay_rate > 0 else None
    weights_initializer_mask = weights_initializer
    biases_initializer = tf.zeros_initializer()

    if norm_mask:
        mask_normalizer_fn = normalizer_fn
        mask_normalizer_params = normalizer_params
    else:
        mask_normalizer_fn = None
        mask_normalizer_params = None

    if deconv:
        if stride == 2:
            ht = upsample(ht, data_format=data_format)
        elif stride != 1:
            raise NotImplementedError

    ht_orig = tf.identity(ht)

    # Normalize hidden state
    with tf.variable_scope('norm_activation_in') as sc:
        if norm_input:
            full_inp = tf.concat([norm_activ(ht), inp], axis=channel_index)
        else:
            full_inp = tf.concat([ht, inp], axis=channel_index)

    # update gate
    rg = conv2d2(full_inp, hidden_depth, 3, sn=sn, stride=1, rate=dilate,
                 data_format=data_format, activation_fn=lrelu,
                 normalizer_fn=mask_normalizer_fn, normalizer_params=mask_normalizer_params,
                 weights_regularizer=regularizer,
                 weights_initializer=weights_initializer_mask,
                 biases_initializer=biases_initializer_mask,
                 scope='update_gate')
    rg = (rg - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True)) / (
        tf.reduce_max(rg, axis=reduce_dim, keep_dims=True) - tf.reduce_min(rg, axis=reduce_dim, keep_dims=True))

    # Input Image conv
    img_new = conv2d2(inp, hidden_depth, 3, sn=sn, stride=1, rate=dilate,
                      data_format=data_format, activation_fn=None,
                      normalizer_fn=None, normalizer_params=None,
                      biases_initializer=biases_initializer,
                      weights_regularizer=regularizer,
                      weights_initializer=weights_initializer)

    ht_plus = ht + rg * img_new
    with tf.variable_scope('norm_activation_merge_1') as sc:
        ht_new_in = norm_activ(ht_plus)

    # new hidden state
    h_new = conv2d2(ht_new_in, filter_depth, 3, sn=sn, stride=1, rate=dilate,
                    data_format=data_format, activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn, normalizer_params=normalizer_params,
                    biases_initializer=biases_initializer,
                    weights_regularizer=regularizer,
                    weights_initializer=weights_initializer)
    h_new = conv2d2(h_new, filter_depth, 3, sn=sn, stride=1, rate=dilate,
                    data_format=data_format, activation_fn=None,
                    normalizer_fn=None, normalizer_params=None,
                    biases_initializer=biases_initializer,
                    weights_regularizer=regularizer,
                    weights_initializer=weights_initializer)

    # new hidden state out
    # linear project for filter depth
    if ht.get_shape().as_list()[channel_index] != filter_depth:
        ht_orig = conv2d2(ht_orig, filter_depth, 1, sn=sn, stride=1,
                          data_format=data_format, activation_fn=None,
                          normalizer_fn=None, normalizer_params=None,
                          biases_initializer=biases_initializer,
                          weights_regularizer=regularizer,
                          weights_initializer=weights_initializer)
    ht_new = ht_orig + h_new

    if not deconv:
        if stride == 2:
            ht_new = mean_pool(ht_new, data_format=data_format)
        elif stride != 1:
            raise NotImplementedError

    return ht_new


def conv2d2(inputs, num_outputs, kernel_size, sn, stride=1, rate=1,
            data_format='NCHW', activation_fn=tf.nn.relu,
            normalizer_fn=None, normalizer_params=None,
            weights_regularizer=None,
            weights_initializer=ly.xavier_initializer(),
            biases_initializer=init_ops.zeros_initializer(),
            biases_regularizer=None,
            reuse=None, scope=None,
            SPECTRAL_NORM_UPDATE_OPS='spectral_norm_update_ops'):
    assert data_format == 'NCHW'
    assert rate == 1
    if data_format == 'NCHW':
        channel_axis = 1
        stride = [1, 1, stride, stride]
        rate = [1, 1, rate, rate]
    else:
        channel_axis = 3
        stride = [1, stride, stride, 1]
        rate = [1, rate, rate, 1]
    input_dim = inputs.get_shape().as_list()[channel_axis]

    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse) as sc:
        inputs = tf.convert_to_tensor(inputs)

        weights = tf.get_variable(name="weights", shape=(kernel_size, kernel_size, input_dim, num_outputs),
                                  initializer=weights_initializer, regularizer=weights_regularizer,
                                  trainable=True, dtype=inputs.dtype.base_dtype)
        # Spectral Normalization
        if sn:
            weights = spectral_normed_weight(
                weights, num_iters=1, update_collection=SPECTRAL_NORM_UPDATE_OPS)

        conv_out = tf.nn.conv2d(
            inputs, weights, strides=stride, padding='SAME', data_format=data_format)

        if biases_initializer is not None:
            biases = tf.get_variable(name='biases', shape=(1, num_outputs, 1, 1),
                                     initializer=biases_initializer, regularizer=biases_regularizer,
                                     trainable=True, dtype=inputs.dtype.base_dtype)
            conv_out += biases

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            conv_out = normalizer_fn(
                conv_out, activation_fn=None, **normalizer_params)

        if activation_fn is not None:
            conv_out = activation_fn(conv_out)

    return conv_out


def mru_conv(x, ht, filter_depth, sn, stride=2, dilate_rate=1,
             num_blocks=5, last_unit=False,
             activation_fn=tf.nn.relu,
             normalizer_fn=None,
             normalizer_params=None,
             weights_initializer=ly.xavier_initializer_conv2d(),
             weight_decay_rate=1e-5,
             unit_num=0, data_format='NCHW'):
    assert len(ht) == num_blocks

    def norm_activ(tensor_in):
        if normalizer_fn is not None:
            _normalizer_params = normalizer_params or {}
            tensor_normed = normalizer_fn(tensor_in, **_normalizer_params)
        else:
            tensor_normed = tf.identity(tensor_in)
        if activation_fn is not None:
            tensor_normed = activation_fn(tensor_normed)

        return tensor_normed

    if dilate_rate != 1:
        stride = 1

    # cell_block = mru_conv_block
    # cell_block = mru_conv_block_v2
    cell_block = functools.partial(mru_conv_block_v3, deconv=False)

    hts_new = []
    inp = x
    with tf.variable_scope('mru_conv_unit_t_%d_layer_0' % unit_num):
        ht_new = cell_block(inp, ht[0], filter_depth, sn=sn, stride=stride,
                            dilate=dilate_rate,
                            activation_fn=activation_fn,
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params,
                            weights_initializer=weights_initializer,
                            data_format=data_format,
                            weight_decay_rate=weight_decay_rate)
        hts_new.append(ht_new)
        inp = ht_new

    for i in range(1, num_blocks):
        if stride == 2:
            ht[i] = mean_pool(ht[i], data_format=data_format)
        with tf.variable_scope('mru_conv_unit_t_%d_layer_%d' % (unit_num, i)):
            ht_new = cell_block(inp, ht[i], filter_depth, sn=sn, stride=1,
                                dilate=dilate_rate,
                                activation_fn=activation_fn,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params,
                                weights_initializer=weights_initializer,
                                data_format=data_format,
                                weight_decay_rate=weight_decay_rate)
            hts_new.append(ht_new)
            inp = ht_new

    if hasattr(cell_block, 'func') and cell_block.func == mru_conv_block_v3 and last_unit:
        with tf.variable_scope('mru_conv_unit_last_norm'):
            hts_new[-1] = norm_activ(hts_new[-1])

    return hts_new


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
