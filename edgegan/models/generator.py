import math

import tensorflow as tf

import edgegan.nn.functional as F
from edgegan import nn


class Generator(object):
    def __init__(self, name, is_train, norm='batch', activation='relu',
                 batch_size=64, output_height=64, output_width=128,
                 input_dim=64, output_dim=3, use_resnet=False):
        print(' [*] Init Generator %s', name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._batch_size = batch_size
        self._output_height = output_height
        self._output_width = output_width
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._use_resnet = use_resnet
        self._reuse = False

    def _conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def __call__(self, z):
        if self._use_resnet:
            return self._resnet(z)
        else:
            return self._convnet(z)

    def _convnet(self, z):
        with tf.variable_scope(self.name, reuse=self._reuse):
            s_h, s_w = self._output_height, self._output_width
            s_h2, s_w2 = self._conv_out_size_same(
                s_h, 2), self._conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self._conv_out_size_same(
                s_h2, 2), self._conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self._conv_out_size_same(
                s_h4, 2), self._conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self._conv_out_size_same(
                s_h8, 2), self._conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            z_ = nn.linear(z, self._input_dim*8 *
                           s_h16*s_w16, name='g_lin_0')
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self._input_dim * 8])
            h0 = nn.activation_fn(nn.norm(
                h0, self._norm), self._activation)

            h1 = nn.deconv_block(h0, [self._batch_size, s_h8, s_w8, self._input_dim*4],
                                 'g_dconv_1', 5, 2, self._is_train,
                                 self._reuse, self._norm, self._activation)

            h2 = nn.deconv_block(h1, [self._batch_size, s_h4, s_w4, self._input_dim*2],
                                 'g_dconv_2', 5, 2, self._is_train,
                                 self._reuse, self._norm, self._activation)

            h3 = nn.deconv_block(h2, [self._batch_size, s_h2, s_w2, self._input_dim],
                                 'g_dconv_3', 5, 2, self._is_train,
                                 self._reuse, self._norm, self._activation)

            h4 = nn.deconv_block(h3, [self._batch_size, s_h, s_w, self._output_dim],
                                 'g_dconv_4', 5, 2, self._is_train,
                                 self._reuse, None, None)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.tanh(h4)

    def _resnet(self, z):
        # return None
        with tf.variable_scope(self.name, reuse=self._reuse):
            s_h, s_w = self._output_height, self._output_width
            s_h2, s_w2 = self._conv_out_size_same(
                s_h, 2), self._conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self._conv_out_size_same(
                s_h2, 2), self._conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self._conv_out_size_same(
                s_h4, 2), self._conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self._conv_out_size_same(
                s_h8, 2), self._conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            z_ = nn.linear(z, self._input_dim*8 *
                           s_h16*s_w16, name='g_lin_resnet_0')
            h0 = nn.activation_fn(nn.norm(
                z_, self._norm), self._activation)
            h0 = tf.reshape(h0, [-1, s_h16, s_w16, self._input_dim * 8])

            h1 = nn.deresidual2(h0, [self._batch_size, s_h8/2, s_w8/2, self._input_dim*4],
                                'g_resnet_1', 3, 1, self._is_train,
                                self._reuse, self._norm, self._activation)
            h1 = nn.upsample2(h1, "NHWC")

            h2 = nn.deresidual2(h1, [self._batch_size, s_h4/2, s_w4/2, self._input_dim*2],
                                'g_resnet_2', 3, 1, self._is_train,
                                self._reuse, self._norm, self._activation)
            h2 = nn.upsample2(h2, "NHWC")

            h3 = nn.deresidual2(h2, [self._batch_size, s_h2/2, s_w2/2, self._input_dim],
                                'g_resnet_3', 3, 1, self._is_train,
                                self._reuse, self._norm, self._activation)
            h3 = nn.upsample2(h3, "NHWC")

            h4 = nn.deresidual2(h3, [self._batch_size, s_h/2, s_w/2, self._output_dim],
                                'g_resnet_4', 3, 1, self._is_train,
                                self._reuse, None, None)
            h4 = nn.upsample2(h4, "NHWC")

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.tanh(h4)
