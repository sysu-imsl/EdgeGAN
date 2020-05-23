import tensorflow as tf

from edgegan import nn


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
                resized_ = nn.mean_pool(
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

            h0 = nn.conv2d2(x_list[-1], 8, kernel_size=7, sn=sn, stride=1, data_format=data_format,
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

            hts_1 = nn.mru_conv(x_list[-1], hts_0,
                                size * 2, sn=sn, stride=2, dilate_rate=1,
                                data_format=data_format, num_blocks=num_blocks,
                                last_unit=False,
                                activation_fn=activation_fn_d,
                                normalizer_fn=normalizer_fn_d,
                                normalizer_params=normalizer_params_d,
                                weights_initializer=weight_initializer,
                                unit_num=1)
            hts_2 = nn.mru_conv(x_list[-2], hts_1,
                                size * 4, sn=sn, stride=2, dilate_rate=1,
                                data_format=data_format, num_blocks=num_blocks,
                                last_unit=False,
                                activation_fn=activation_fn_d,
                                normalizer_fn=normalizer_fn_d,
                                normalizer_params=normalizer_params_d,
                                weights_initializer=weight_initializer,
                                unit_num=2)
            hts_3 = nn.mru_conv(x_list[-3], hts_2,
                                size * 8, sn=sn, stride=2, dilate_rate=1,
                                data_format=data_format, num_blocks=num_blocks,
                                last_unit=False,
                                activation_fn=activation_fn_d,
                                normalizer_fn=normalizer_fn_d,
                                normalizer_params=normalizer_params_d,
                                weights_initializer=weight_initializer,
                                unit_num=3)
            hts_4 = nn.mru_conv(x_list[-4], hts_3,
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
            disc = nn.conv2d2(img, output_dim, kernel_size=1, sn=sn, stride=1, data_format=data_format,
                              activation_fn=None, normalizer_fn=None,
                              weights_initializer=weight_initializer)

            # classification end
            img = tf.reduce_mean(img, axis=(
                2, 3) if data_format == 'NCHW' else (1, 2))
            logits = nn.fully_connected(img, num_classes, sn=sn, activation_fn=None,
                                        normalizer_fn=None, SPECTRAL_NORM_UPDATE_OPS=self.SPECTRAL_NORM_UPDATE_OPS)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.name)
        return disc, tf.nn.sigmoid(logits), logits
