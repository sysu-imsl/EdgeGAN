import tensorflow as tf

import edgegan.nn.functional as F
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
                E = nn.conv_block(E, n, 'e_convnet_{}_{}'.format(n, i), 4,
                                        2, self._is_train, self._reuse,
                                        norm=self._norm if i else None,
                                        activation=self._activation)
            E = F.flatten(E)
            mu = nn.mlp(E, self._latent_dim, 'FC8_mu', self._is_train,
                        self._reuse, norm=None, activation=None)
            log_sigma = nn.mlp(E, self._latent_dim, 'FC8_sigma',
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
            E = nn.conv_block(E, 64, 'e_resnet_{}_{}'.format(64, 0), 4, 2,
                              self._is_train, self._reuse, norm=None,
                              activation=self._activation, bias=True)
            for i, n in enumerate(num_filters):
                E = nn.residual(E, n, 'e_resnet_{}_{}'.format(n, i + 1),
                                      self._is_train, self._reuse,
                                      norm=self._norm, bias=True)
                E = tf.nn.avg_pool(E, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
            E = nn.activation_fn(E, 'relu')
            E = tf.nn.avg_pool(E, [1, 8, 8, 1], [1, 8, 8, 1], 'SAME')
            E = F.flatten(E)
            mu = nn.mlp(E, self._latent_dim, 'FC8_mu', self._is_train,
                        self._reuse, norm=None, activation=None)
            log_sigma = nn.mlp(E, self._latent_dim, 'FC8_sigma',
                               self._is_train, self._reuse, norm=None,
                               activation=None)

            z = mu + tf.random_normal(shape=tf.shape(self._latent_dim)) \
                * tf.exp(log_sigma)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)
            return z, mu, log_sigma
