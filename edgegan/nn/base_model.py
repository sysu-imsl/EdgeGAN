# -*- coding:utf8 -*-
# defination of encoder, generator, discriminators

import tensorflow as tf
import tensorflow.contrib.layers as ly
import networks
import math
from edgegan import nn


class Encoder(object):
    def __init__(self, name, is_train, norm='batch', activation='relu',
                 image_size=128, latent_dim=8,
                 use_resnet=True):
        print(' [*] Init Encoder %s', name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._image_size = image_size
        self._latent_dim = latent_dim
        self._use_resnet = use_resnet
        self._reuse = False

    def __call__(self, input):
        if self._use_resnet:
            return self._resnet(input)
        else:
            return self._convnet(input)

    def _convnet(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            num_filters = [64, 128, 256, 512, 512, 512, 512]
            if self._image_size == 256:
                num_filters.append(512)

            E = input
            for i, n in enumerate(num_filters):
                E = networks.conv_block(E, n, 'e_convnet_{}_{}'.format(n, i), 4,
                                        2, self._is_train, self._reuse,
                                        norm=self._norm if i else None,
                                        activation=self._activation)
            E = networks.flatten(E)
            mu = networks.mlp(E, self._latent_dim, 'FC8_mu', self._is_train,
                              self._reuse, norm=None, activation=None)
            log_sigma = networks.mlp(E, self._latent_dim, 'FC8_sigma',
                                     self._is_train, self._reuse,
                                     norm=None, activation=None)

            z = mu + tf.random_normal(shape=tf.shape(self._latent_dim)) \
                * tf.exp(log_sigma)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)
            return z, mu, log_sigma

    def _resnet(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            num_filters = [128, 256, 512, 512]
            if self._image_size == 256:
                num_filters.append(512)

            E = input
            E = networks.conv_block(E, 64, 'e_resnet_{}_{}'.format(64, 0), 4, 2,
                                    self._is_train, self._reuse, norm=None,
                                    activation=self._activation, bias=True)
            for i, n in enumerate(num_filters):
                E = networks.residual(E, n, 'e_resnet_{}_{}'.format(n, i + 1),
                                      self._is_train, self._reuse,
                                      norm=self._norm, bias=True)
                E = tf.nn.avg_pool(E, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            E = networks._activation(E, 'relu')
            E = tf.nn.avg_pool(E, [1, 8, 8, 1], [1, 8, 8, 1], 'SAME')
            E = networks.flatten(E)
            mu = networks.mlp(E, self._latent_dim, 'FC8_mu', self._is_train,
                              self._reuse, norm=None, activation=None)
            log_sigma = networks.mlp(E, self._latent_dim, 'FC8_sigma',
                                     self._is_train, self._reuse, norm=None,
                                     activation=None)

            z = mu + tf.random_normal(shape=tf.shape(self._latent_dim)) \
                * tf.exp(log_sigma)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)
            return z, mu, log_sigma


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
            z_ = networks._linear(z, self._input_dim*8 *
                                  s_h16*s_w16, name='g_lin_0')
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self._input_dim * 8])
            h0 = networks._activation(networks._norm(
                h0, self._norm), self._activation)

            h1 = networks.deconv_block(h0, [self._batch_size, s_h8, s_w8, self._input_dim*4],
                                       'g_dconv_1', 5, 2, self._is_train,
                                       self._reuse, self._norm, self._activation)

            h2 = networks.deconv_block(h1, [self._batch_size, s_h4, s_w4, self._input_dim*2],
                                       'g_dconv_2', 5, 2, self._is_train,
                                       self._reuse, self._norm, self._activation)

            h3 = networks.deconv_block(h2, [self._batch_size, s_h2, s_w2, self._input_dim],
                                       'g_dconv_3', 5, 2, self._is_train,
                                       self._reuse, self._norm, self._activation)

            h4 = networks.deconv_block(h3, [self._batch_size, s_h, s_w, self._output_dim],
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
            z_ = networks._linear(z, self._input_dim*8 *
                                  s_h16*s_w16, name='g_lin_resnet_0')
            h0 = networks._activation(networks._norm(
                z_, self._norm), self._activation)
            h0 = tf.reshape(h0, [-1, s_h16, s_w16, self._input_dim * 8])

            h1 = networks.deresidual2(h0, [self._batch_size, s_h8/2, s_w8/2, self._input_dim*4],
                                      'g_resnet_1', 3, 1, self._is_train,
                                      self._reuse, self._norm, self._activation)
            h1 = networks.upsample2(h1, "NHWC")

            h2 = networks.deresidual2(h1, [self._batch_size, s_h4/2, s_w4/2, self._input_dim*2],
                                      'g_resnet_2', 3, 1, self._is_train,
                                      self._reuse, self._norm, self._activation)
            h2 = networks.upsample2(h2, "NHWC")

            h3 = networks.deresidual2(h2, [self._batch_size, s_h2/2, s_w2/2, self._input_dim],
                                      'g_resnet_3', 3, 1, self._is_train,
                                      self._reuse, self._norm, self._activation)
            h3 = networks.upsample2(h3, "NHWC")

            h4 = networks.deresidual2(h3, [self._batch_size, s_h/2, s_w/2, self._output_dim],
                                      'g_resnet_4', 3, 1, self._is_train,
                                      self._reuse, None, None)
            h4 = networks.upsample2(h4, "NHWC")

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.tanh(h4)


class Discriminator(object):
    def __init__(self, name, is_train, norm='batch', activation='lrelu',
                 num_filters=64, use_resnet=False):
        print(' [*] Init Discriminator %s', name)
        self._num_filters = num_filters
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._use_resnet = use_resnet
        self._reuse = False

    def __call__(self, input, reuse=False):
        if self._use_resnet:
            return self._resnet(input)
        else:
            return self._convnet(input)

    def _resnet(self, input):
        # return None
        with tf.variable_scope(self.name, reuse=self._reuse):
            D = networks.residual2(input, self._num_filters, 'd_resnet_0', 3, 1,
                                   self._is_train, self._reuse, norm=None,
                                   activation=self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = networks.residual2(D, self._num_filters*2, 'd_resnet_1', 3, 1,
                                   self._is_train, self._reuse, self._norm,
                                   self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = networks.residual2(D, self._num_filters*4, 'd_resnet_3', 3, 1,
                                   self._is_train, self._reuse, self._norm,
                                   self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = networks.residual2(D, self._num_filters*8, 'd_resnet_4', 3, 1,
                                   self._is_train, self._reuse, self._norm,
                                   self._activation)
            D = tf.nn.avg_pool(D, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            D = networks._activation(D, self._activation)
            D = tf.nn.avg_pool(D, [1, 8, 8, 1], [1, 8, 8, 1], 'SAME')

            D = networks._linear(tf.reshape(D, [input.get_shape()[0], -1]), 1,
                                 name='d_linear_resnet_5')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.sigmoid(D), D

    def _convnet(self, input):

        with tf.variable_scope(self.name, reuse=self._reuse):
            D = networks.conv_block(input, self._num_filters, 'd_conv_0', 4, 2,
                                    self._is_train, self._reuse, norm=None,
                                    activation=self._activation)
            D = networks.conv_block(D, self._num_filters*2, 'd_conv_1', 4, 2,
                                    self._is_train, self._reuse, self._norm,
                                    self._activation)
            D = networks.conv_block(D, self._num_filters*4, 'd_conv_3', 4, 2,
                                    self._is_train, self._reuse, self._norm,
                                    self._activation)
            D = networks.conv_block(D, self._num_filters*8, 'd_conv_4', 4, 2,
                                    self._is_train, self._reuse, self._norm,
                                    self._activation)

            D = networks._linear(tf.reshape(D, [input.get_shape()[0], -1]), 1,
                                 name='d_linear_5')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.sigmoid(D), D


class Classifier(object):
    def __init__(self, name, SPECTRAL_NORM_UPDATE_OPS):
        print(' [*] Init Discriminator %s', name)
        self.name = name
        self.SPECTRAL_NORM_UPDATE_OPS = SPECTRAL_NORM_UPDATE_OPS

    def __call__(self, x, num_classes, labels=None, reuse=False, data_format='NCHW'):
        assert data_format == 'NCHW'
        size = 64
        num_blocks = 1
        resize_func = tf.image.resize_bilinear
        sn = True

        if data_format == 'NCHW':
            channel_axis = 1
        else:
            channel_axis = 3
        if type(x) is list:
            x = x[-1]

        if data_format == 'NCHW':
            x_list = []
            resized_ = x
            x_list.append(resized_)

            for i in range(5):
                resized_ = networks.mean_pool(
                    resized_, data_format=data_format)
                x_list.append(resized_)
            x_list = x_list[::-1]
        else:
            raise NotImplementedError

        output_dim = 1

        activation_fn_d = nn.prelu
        normalizer_fn_d = None
        normalizer_params_d = None
        weight_initializer = tf.random_normal_initializer(0, 0.02)

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            h0 = networks.conv2d2(x_list[-1], 8, kernel_size=7, sn=sn, stride=1, data_format=data_format,
                                  activation_fn=activation_fn_d,
                                  normalizer_fn=normalizer_fn_d,
                                  normalizer_params=normalizer_params_d,
                                  weights_initializer=weight_initializer)

            # Initial memory state
            hidden_state_shape = h0.get_shape().as_list()
            batch_size = hidden_state_shape[0]
            hidden_state_shape[0] = 1
            hts_0 = [h0]
            for i in range(1, num_blocks):
                h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                             initializer=tf.zeros_initializer()), [batch_size, 1, 1, 1])
                hts_0.append(h0)

            hts_1 = networks.mru_conv(x_list[-1], hts_0,
                                      size * 2, sn=sn, stride=2, dilate_rate=1,
                                      data_format=data_format, num_blocks=num_blocks,
                                      last_unit=False,
                                      activation_fn=activation_fn_d,
                                      normalizer_fn=normalizer_fn_d,
                                      normalizer_params=normalizer_params_d,
                                      weights_initializer=weight_initializer,
                                      unit_num=1)
            hts_2 = networks.mru_conv(x_list[-2], hts_1,
                                      size * 4, sn=sn, stride=2, dilate_rate=1,
                                      data_format=data_format, num_blocks=num_blocks,
                                      last_unit=False,
                                      activation_fn=activation_fn_d,
                                      normalizer_fn=normalizer_fn_d,
                                      normalizer_params=normalizer_params_d,
                                      weights_initializer=weight_initializer,
                                      unit_num=2)
            hts_3 = networks.mru_conv(x_list[-3], hts_2,
                                      size * 8, sn=sn, stride=2, dilate_rate=1,
                                      data_format=data_format, num_blocks=num_blocks,
                                      last_unit=False,
                                      activation_fn=activation_fn_d,
                                      normalizer_fn=normalizer_fn_d,
                                      normalizer_params=normalizer_params_d,
                                      weights_initializer=weight_initializer,
                                      unit_num=3)
            hts_4 = networks.mru_conv(x_list[-4], hts_3,
                                      size * 12, sn=sn, stride=2, dilate_rate=1,
                                      data_format=data_format, num_blocks=num_blocks,
                                      last_unit=True,
                                      activation_fn=activation_fn_d,
                                      normalizer_fn=normalizer_fn_d,
                                      normalizer_params=normalizer_params_d,
                                      weights_initializer=weight_initializer,
                                      unit_num=4)

            img = hts_4[-1]
            img_shape = img.get_shape().as_list()

            # discriminator end
            disc = networks.conv2d2(img, output_dim, kernel_size=1, sn=sn, stride=1, data_format=data_format,
                                    activation_fn=None, normalizer_fn=None,
                                    weights_initializer=weight_initializer)

            # classification end
            img = tf.reduce_mean(img, axis=(
                2, 3) if data_format == 'NCHW' else (1, 2))
            logits = networks.fully_connected(img, num_classes, sn=sn, activation_fn=None,
                                              normalizer_fn=None, SPECTRAL_NORM_UPDATE_OPS=self.SPECTRAL_NORM_UPDATE_OPS)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)
        return disc, tf.nn.sigmoid(logits), logits


class Discriminator_patch(object):
    def __init__(self, name, is_train, norm='batch', activation='lrelu',
                 num_filters=64, use_resnet=False):
        print(' [*] Init Discriminator %s', name)
        self._num_filters = num_filters
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._activation = activation
        self._use_resnet = use_resnet
        self._reuse = False

    def __call__(self, input, reuse=False):
        if self._use_resnet:
            return self._resnet(input)
        else:
            return self._convnet(input)

    def _resnet(self, input):
        return None

    def _convnet(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            D = networks.conv_block(input, self._num_filters, 'd_patch_conv_0', 4, 2,
                                    self._is_train, self._reuse, norm=None,
                                    activation=self._activation)
            D = networks.conv_block(D, self._num_filters*2, 'd_patch_conv_1', 4, 1,
                                    self._is_train, self._reuse, self._norm,
                                    self._activation)
            D = networks.conv_block(D, 1, 'd_patch_conv_2', 4, 1,
                                    self._is_train, self._reuse, self._norm,
                                    self._activation)
            D = networks._linear(tf.reshape(D, [input.get_shape()[0], -1]), 1,
                                 name='d_patch_linear_3')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)

            return tf.nn.sigmoid(D), D


class patchGAN_D(object):
    def __init__(self, name, norm='batch',
                 num_filters=64):
        print(' [*] Init Discriminator %s', name)
        self._num_filters = num_filters
        self.name = name
        self._norm = norm
        self._reuse = False

    def __call__(self, discrim_inputs, discrim_targets):
        return self.create_discriminator(discrim_inputs, discrim_targets)

    def discrim_conv(self, batch_input, out_channels, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [
                              1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid",
                                kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def lrelu(self, x, a):
        with tf.name_scope("lrelu"):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: a*x/2 - a*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def batchnorm(self, inputs):
        return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                             gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

    def instancenorm(self, inputs):
        with tf.variable_scope('instance_norm'):
            eps = 1e-5
            mean, sigma = tf.nn.moments(inputs, [1, 2], keep_dims=True)
            normalized = (inputs - mean) / (tf.sqrt(sigma) + eps)
            return normalized

    def create_discriminator(self, discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        with tf.variable_scope(self.name, reuse=self._reuse):
            convolved = self.discrim_conv(input, self._num_filters, stride=2)
            rectified = self.lrelu(convolved, 0.2)
            layers.append(rectified)
            for i in range(n_layers):
                out_channels = self._num_filters * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = self.discrim_conv(
                    layers[-1], out_channels, stride=stride)
                if self._norm == "batch":
                    normalized = self.batchnorm(convolved)
                elif self._norm == "instance":
                    normalized = self.instancenorm(convolved)
                rectified = self.lrelu(normalized, 0.2)
                layers.append(rectified)

            convolved = self.discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        self._reuse = True
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          self.name)
        return layers[-1], convolved
