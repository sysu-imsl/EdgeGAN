# -*- coding:utf8 -*-
# defination of dcgan

from __future__ import division
from __future__ import print_function
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from base_model import Encoder, Generator, Discriminator, Discriminator2, Discriminator_patch, patchGAN_D
import base_model
import networks
import utils

import sys
import pickle

# reload(sys)
# sys.setdefaultencoding('utf8')


def save_layer(name, tensor):
    return tf.keras.layers.Lambda(save_tensor(name))(tensor)


def save_tensor(name):
    output_folder = 'checksum'

    @tf.function
    def wrapper(x):
        # with open(os.path.join(output_folder, name), 'w') as f:
        tf.print(x, output_stream='file://' +
                 os.path.join(output_folder, name))
        return x
    return wrapper


class DCGAN(object):
    def __init__(self, sess, config,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3):
        """

        Args:
        sess: TensorFlow session
        batch_size: The size of batch. Should be specified before training.
        z_dim: (optional) Dimension of dim for Z. [100]
        gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.config = config

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

    def build_model1(self):
        os.system('rm checksum/*')
        # Define models
        if self.config.model is "old":
            if self.config.G_num == 2:
                self.generator1 = Generator('G1', is_train=True,
                                            norm=self.config.G_norm,
                                            batch_size=self.config.batch_size,
                                            output_height=self.config.output_height,
                                            output_width=int(
                                                self.config.output_width/2),
                                            input_dim=self.gf_dim,
                                            output_dim=self.c_dim,
                                            use_resnet=self.config.if_resnet_g)
                self.generator2 = Generator('G2', is_train=True,
                                            norm=self.config.G_norm,
                                            batch_size=self.config.batch_size,
                                            output_height=self.config.output_height,
                                            output_width=int(
                                                self.config.output_width/2),
                                            input_dim=self.gf_dim,
                                            output_dim=self.c_dim,
                                            use_resnet=self.config.if_resnet_g)
            else:
                self.generator = Generator('G', is_train=True,
                                           norm=self.config.G_norm,
                                           batch_size=self.config.batch_size,
                                           output_height=self.config.output_height,
                                           output_width=self.config.output_width,
                                           input_dim=self.gf_dim,
                                           output_dim=self.c_dim,
                                           use_resnet=self.config.if_resnet_g)
        else:
            self.generator = base_model.generator_model

        # classifier
        if self.config.if_focal_loss:
            self.discriminator2 = Discriminator2(
                'D2', self.config.SPECTRAL_NORM_UPDATE_OPS)

        # origin discrminator
        if self.config.use_D_origin is True:
            if self.config.model is "old":
                self.discriminator = Discriminator('D', is_train=True,
                                                   norm=self.config.D_norm,
                                                   num_filters=self.df_dim,
                                                   use_resnet=self.config.if_resnet_d)
            else:
                self.discriminator = base_model.discriminator_model

        # multi-scale discriminators
        if self.config.use_D_patch is True:
            self.discriminator_patch = Discriminator_patch('D_patch', is_train=True,
                                                           norm=self.config.D_patch_norm,
                                                           num_filters=self.df_dim,
                                                           use_resnet=False)

        if self.config.use_D_patch2 is True:
            self.discriminator_patch2 = Discriminator('D_patch2', is_train=True,
                                                      norm=self.config.D_norm,
                                                      num_filters=self.df_dim,
                                                      use_resnet=self.config.if_resnet_d)

        if self.config.use_D_patch2_2 is True:
            self.discriminator_patch2_2 = Discriminator('D_patch2_2', is_train=True,
                                                        norm=self.config.D_norm,
                                                        num_filters=self.df_dim,
                                                        use_resnet=self.config.if_resnet_d)

        if self.config.use_D_patch3 is True:
            self.discriminator_patch3 = Discriminator('D_patch3', is_train=True,
                                                      norm=self.config.D_norm,
                                                      num_filters=self.df_dim,
                                                      use_resnet=self.config.if_resnet_d)

        if self.config.G_num == 2 and self.config.use_patchGAN_D_full == True:
            self.discriminator_patchGAN = patchGAN_D(
                "D_patchGAN", norm=self.config.patchGAN_D_norm, num_filters=64)

        if self.config.E_stage1:
            if self.config.if_focal_loss:
                self.encoder = Encoder('E', is_train=True,
                                       norm=self.config.E_norm,
                                       image_size=self.config.input_height,
                                       latent_dim=self.z_dim,
                                       use_resnet=self.config.if_resnet_e)
                if self.config.E2_stage1:
                    self.encoder2 = Encoder('E2', is_train=True,
                                            norm=self.config.E_norm,
                                            image_size=self.config.input_height,
                                            latent_dim=self.config.num_classes,
                                            use_resnet=self.config.if_resnet_e)
            else:
                self.encoder = Encoder('E', is_train=True,
                                       norm=self.config.E_norm,
                                       image_size=self.config.input_height,
                                       latent_dim=self.z_dim,
                                       use_resnet=self.config.if_resnet_e)

        # define inputs
        if self.config.crop:
            self.image_dims = [self.config.output_height, self.config.output_width,
                               self.c_dim]
        else:
            self.image_dims = [self.config.input_height, self.config.input_width,
                               self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.config.batch_size] + self.image_dims, name='real_images')

        if self.config.if_focal_loss:
            self.z = tf.placeholder(
                tf.float32, [None, self.z_dim+1], name='z')
        else:
            self.z = tf.placeholder(
                tf.float32, [None, self.z_dim], name='z')

        if self.config.if_focal_loss:
            self.class_onehot = tf.one_hot(tf.cast(
                self.z[:, -1], dtype=tf.int32), self.config.num_classes, on_value=1., off_value=0., dtype=tf.float32)
            self.z_onehot = tf.concat(
                [self.z[:, 0:self.z_dim], self.class_onehot], 1)
            if self.config.G_num == 2:
                self.G1 = self.generator1(self.z_onehot)
                self.G2 = self.generator2(self.z_onehot)
            else:
                self.G = self.generator(self.z_onehot)
        else:
            if self.config.G_num == 2:
                self.G1 = self.generator1(self.z)
                self.G2 = self.generator2(self.z)
            else:
                self.G = self.generator(self.z)
        if self.config.if_focal_loss:
            if self.image_dims[0] == self.image_dims[1]:
                _, _, self.D_logits2 = self.discriminator2(
                    tf.transpose(self.inputs, [0, 3, 1, 2]),
                    num_classes=self.config.num_classes,
                    labels=self.z[:, -1],
                    reuse=False, data_format='NCHW')
                _, _, self.D_logits2_ = self.discriminator2(
                    tf.transpose(self.G, [0, 3, 1, 2]),
                    num_classes=self.config.num_classes,
                    labels=self.z[:, -1],
                    reuse=True, data_format='NCHW')
            else:
                if self.config.classifier_inputForm is "image":
                    _, _, self.D_logits2 = self.discriminator2(
                        tf.transpose(self.inputs[:, :, int(
                            self.image_dims[1]/2):, :], [0, 3, 1, 2]),
                        num_classes=self.config.num_classes,
                        labels=self.z[:, -1],
                        reuse=False, data_format='NCHW')
                    if self.config.G_num == 2:
                        _, _, self.D_logits2_ = self.discriminator2(
                            tf.transpose(self.G2, [0, 3, 1, 2]),
                            num_classes=self.config.num_classes,
                            labels=self.z[:, -1],
                            reuse=True, data_format='NCHW')
                    else:
                        _, _, self.D_logits2_ = self.discriminator2(
                            tf.transpose(
                                self.G[:, :, int(self.image_dims[1]/2):, :], [0, 3, 1, 2]),
                            num_classes=self.config.num_classes,
                            labels=self.z[:, -1],
                            reuse=True, data_format='NCHW')
                elif self.config.classifier_inputForm is "edge":
                    _, _, self.D_logits2 = self.discriminator2(
                        tf.transpose(self.inputs[:, :, 0:int(
                            self.image_dims[1]/2), :], [0, 3, 1, 2]),
                        num_classes=self.config.num_classes,
                        labels=self.z[:, -1],
                        reuse=False, data_format='NCHW')
                    _, _, self.D_logits2_ = self.discriminator2(
                        tf.transpose(
                            self.G[:, :, 0:int(self.image_dims[1]/2), :], [0, 3, 1, 2]),
                        num_classes=self.config.num_classes,
                        labels=self.z[:, -1],
                        reuse=True, data_format='NCHW')

        if self.config.use_D_origin:
            if self.config.originD_inputForm == "concat_w":
                self.D, self.D_logits = self.discriminator(self.inputs)
                if self.config.G_num == 2:
                    self.G_all = tf.concat([self.G1, self.G2], 2)
                    self.D_, self.D_logits_ = self.discriminator(
                        self.G_all, reuse=True)
                else:
                    self.D_, self.D_logits_ = self.discriminator(
                        self.G, reuse=True)
            elif self.config.originD_inputForm == "concat_n":
                left_in = self.inputs[:, :, 0:int(
                    self.config.output_width / 2), :]
                right_in = self.inputs[:, :, int(
                    self.config.output_width / 2):self.config.output_width, :]
                self.concat_in_n = tf.concat([left_in, right_in], 3)
                self.D, self.D_logits = self.discriminator(self.concat_in_n)
                if self.config.G_num == 2:
                    self.G_all = tf.concat([self.G1, self.G2], 2)
                    self.concat_out_n = tf.concat([self.G1, self.G2], 3)
                    self.D_, self.D_logits_ = self.discriminator(
                        self.concat_out_n, reuse=True)
                else:
                    left_G = self.G[:, :, 0:int(
                        self.config.output_width / 2), :]
                    right_G = self.G[:, :, int(
                        self.config.output_width / 2):self.config.output_width, :]
                    self.concat_out_n = tf.concat([left_G, right_G], 3)
                    self.D_, self.D_logits_ = self.discriminator(
                        self.concat_out_n, reuse=True)

        if self.config.use_D_patch:
            self.patch_D, self.patch_D_logits = self.discriminator_patch(
                self.inputs)
            if self.config.G_num == 2:
                self.G_all = tf.concat([self.G1, self.G2], 2)
                self.patch_D_, self.patch_D_logits_ = self.discriminator_patch(
                    self.G_all, reuse=True)
            else:
                self.patch_D_, self.patch_D_logits_ = self.discriminator_patch(
                    self.G, reuse=True)

        if self.config.use_D_patch2:
            if self.config.conditional_D2 == "full_concat_w":
                #   TODO: Resize the input
                self.resized_inputs = tf.image.resize_images(
                    self.inputs, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2 * 2], method=2)
                self.patch2_D, self.patch2_D_logits = self.discriminator_patch2(
                    self.resized_inputs)
                self.resized_inputs_image = self.resized_inputs

                self.resized_G = tf.image.resize_images(self.G, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2 * 2],
                                                        method=2)
                self.patch2_D_, self.patch2_D_logits_ = self.discriminator_patch2(
                    self.resized_G, reuse=True)
                self.resized_G_image = self.resized_G
            elif self.config.conditional_D2 == "full_concat_n":
                left_i = self.inputs[:, :, 0:int(
                    self.config.output_width/2), :]
                right_i = self.inputs[:, :, int(
                    self.config.output_width/2):self.config.output_width, :]
                left_i = tf.image.resize_images(
                    left_i, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2], method=2)
                right_i = tf.image.resize_images(right_i, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2],
                                                 method=2)
                self.resized_inputs_image = right_i
                self.resized_inputs = tf.concat([left_i, right_i], 3)
                self.patch2_D, self.patch2_D_logits = self.discriminator_patch2(
                    self.resized_inputs)

                left_G = self.G[:, :, 0:int(self.config.output_width / 2), :]
                right_G = self.G[:, :, int(
                    self.config.output_width / 2):self.config.output_width, :]
                left_G = tf.image.resize_images(left_G, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2],
                                                method=2)
                right_G = tf.image.resize_images(right_G, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2],
                                                 method=2)
                self.resized_G_image = right_G
                self.resized_G = tf.concat([left_G, right_G], 3)
                self.patch2_D_, self.patch2_D_logits_ = self.discriminator_patch2(
                    self.resized_G, reuse=True)
            elif self.config.conditional_D2 == "single_right":
                right_i = self.inputs[:, :, int(
                    self.config.output_width / 2):self.config.output_width, :]
                right_i = tf.image.resize_images(right_i, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2],
                                                 method=2)
                self.resized_inputs = right_i
                self.resized_inputs_image = self.resized_inputs
                self.patch2_D, self.patch2_D_logits = self.discriminator_patch2(
                    self.resized_inputs)

                if self.config.G_num == 2:
                    self.resized_G2_p2 = tf.image.resize_images(self.G2,
                                                                [self.config.sizeOfIn_patch2,
                                                                 self.config.sizeOfIn_patch2],
                                                                method=2)
                    self.patch2_D_, self.patch2_D_logits_ = self.discriminator_patch2(
                        self.resized_G2_p2, reuse=True)
                else:
                    right_G = self.G[:, :, int(
                        self.config.output_width / 2):self.config.output_width, :]
                    right_G = tf.image.resize_images(right_G, [self.config.sizeOfIn_patch2, self.config.sizeOfIn_patch2],
                                                     method=2)
                    self.resized_G = right_G
                    self.resized_G_image = self.resized_G
                    self.patch2_D_, self.patch2_D_logits_ = self.discriminator_patch2(
                        self.resized_G, reuse=True)

        if self.config.use_D_patch2_2 == True:
            right_i = self.inputs[:, :, int(
                self.config.output_width / 2):self.config.output_width, :]
            right_i = tf.image.resize_images(right_i, [self.config.sizeOfIn_patch2_2, self.config.sizeOfIn_patch2_2],
                                             method=2)
            self.resized_inputs_p2_2 = right_i
            self.resized_inputs_p2_2_image = self.resized_inputs_p2_2
            self.patch2_2_D, self.patch2_2_D_logits = self.discriminator_patch2_2(
                self.resized_inputs_p2_2)

            if self.config.G_num == 2:
                self.resized_G2_p2_2 = tf.image.resize_images(self.G2,
                                                              [self.config.sizeOfIn_patch2_2,
                                                               self.config.sizeOfIn_patch2_2],
                                                              method=2)
                self.patch2_2_D_, self.patch2_2_D_logits_ = self.discriminator_patch2_2(
                    self.resized_G2_p2_2, reuse=True)

        if self.config.use_D_patch3:
            #   TODO: Resize the input
            if self.config.conditional_D3 == "full_concat_w":
                self.resized_inputs_p3 = tf.image.resize_images(
                    self.inputs, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3 * 2], method=2)
                self.patch3_D, self.patch3_D_logits = self.discriminator_patch3(
                    self.resized_inputs_p3)
                self.resized_inputs_p3_image = self.resized_inputs_p3

                self.resized_G_p3 = tf.image.resize_images(self.G, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3 * 2],
                                                           method=2)
                self.patch3_D_, self.patch3_D_logits_ = self.discriminator_patch3(
                    self.resized_G_p3, reuse=True)
                self.resized_G_p3_image = self.resized_G_p3
            elif self.config.conditional_D3 == "full_concat_n":
                left_i = self.inputs[:, :, 0:int(
                    self.config.output_width/2), :]
                right_i = self.inputs[:, :, int(
                    self.config.output_width/2):self.config.output_width, :]
                left_i = tf.image.resize_images(
                    left_i, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3], method=2)
                right_i = tf.image.resize_images(right_i, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3],
                                                 method=2)
                self.resized_inputs_p3_image = right_i
                self.resized_inputs_p3 = tf.concat([left_i, right_i], 3)
                self.patch3_D, self.patch3_D_logits = self.discriminator_patch3(
                    self.resized_inputs_p3)

                left_G = self.G[:, :, 0:int(self.config.output_width / 2), :]
                right_G = self.G[:, :, int(
                    self.config.output_width / 2):self.config.output_width, :]
                left_G = tf.image.resize_images(left_G, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3],
                                                method=2)
                right_G = tf.image.resize_images(right_G, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3],
                                                 method=2)
                self.resized_G_p3_image = right_G
                self.resized_G_p3 = tf.concat([left_G, right_G], 3)
                self.patch3_D_, self.patch3_D_logits_ = self.discriminator_patch3(
                    self.resized_G_p3, reuse=True)
            elif self.config.conditional_D3 == "single_right":
                left_i = self.inputs[:, :, 0:int(
                    self.config.output_width / 2), :]
                left_i = tf.image.resize_images(left_i, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3],
                                                method=2)

                right_i = self.inputs[:, :, int(
                    self.config.output_width / 2):self.config.output_width, :]
                right_i = tf.image.resize_images(right_i, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3],
                                                 method=2)
                if self.config.G_num == 2:
                    self.resized_inputs_p3 = left_i
                else:
                    self.resized_inputs_p3 = right_i

                self.resized_inputs_p3_image = self.resized_inputs_p3
                self.patch3_D, self.patch3_D_logits = self.discriminator_patch3(
                    self.resized_inputs_p3)

                if self.config.G_num == 2:
                    self.resized_G1_p3 = tf.image.resize_images(self.G1,
                                                                [self.config.sizeOfIn_patch3,
                                                                    self.config.sizeOfIn_patch3],
                                                                method=2)
                    self.patch3_D_, self.patch3_D_logits_ = self.discriminator_patch3(
                        self.resized_G1_p3, reuse=True)
                else:
                    right_G = self.G[:, :, int(
                        self.config.output_width / 2):self.config.output_width, :]
                    right_G = tf.image.resize_images(right_G, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3],
                                                     method=2)
                    self.resized_G_p3 = right_G
                    self.resized_G_p3_image = self.resized_G_p3
                    self.patch3_D_, self.patch3_D_logits_ = self.discriminator_patch3(
                        self.resized_G_p3, reuse=True)

        if self.config.E_stage1:
            if self.config.G_num == 2:
                z_recon, z_recon_mu, z_recon_log_sigma = self.encoder(self.G1)
                if self.config.if_focal_loss and self.config.E2_stage1:
                    class_recon, _, _ = self.encoder2(self.G1)
            else:
                self.G_left = self.G[0:self.config.batch_size, 0:self.image_dims[0], 0:int(self.image_dims[1] / 2),
                                     0:self.image_dims[2]]
                z_recon, z_recon_mu, z_recon_log_sigma = self.encoder(
                    self.G_left)

                if self.config.if_focal_loss and self.config.E2_stage1:
                    class_recon, _, _ = self.encoder2(self.G_left)

        # define loss
        if self.config.G_num == 2 and self.config.use_patchGAN_D_full == True:
            predict_fake, predict_fake_logit = self.discriminator_patchGAN(
                self.G1, self.G2)
            predict_real, predict_real_logit = self.discriminator_patchGAN(self.inputs[:, :, 0:int(self.config.output_width / 2), :],
                                                                           self.inputs[:, :,
                                                                                       int(self.config.output_width / 2):self.config.output_width, :])
            if self.config.patchGAN_loss == "gpwgan":
                self.d_loss_patchGAN = tf.reduce_mean(
                    predict_fake_logit - predict_real_logit)

                self.concat_out_n = tf.concat([self.G1, self.G2], 3)
                left_in = self.inputs[:, :, 0:int(
                    self.config.output_width / 2), :]
                right_in = self.inputs[:, :, int(
                    self.config.output_width / 2):self.config.output_width, :]
                self.concat_in_n = tf.concat([left_in, right_in], 3)
                alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
                alpha = alpha_dist.sample((self.config.batch_size, 1, 1, 1))
                interpolated1 = left_in + alpha * (self.G1 - left_in)
                interpolated2 = right_in + alpha * (self.G2 - right_in)
                interpolated = tf.concat([interpolated1, interpolated2], 3)
                inte_logit_tmp1, inte_logit_tmp2 = self.discriminator_patchGAN(
                    interpolated1, interpolated2)
                inte_logit_tmp1_tmp = tf.reduce_mean(tf.reduce_mean(
                    inte_logit_tmp1, axis=2, keep_dims=False), 1)
                inte_logit_tmp2_tmp = tf.reduce_mean(tf.reduce_mean(
                    inte_logit_tmp2, axis=2, keep_dims=False), 1)
                inte_logit = tuple([inte_logit_tmp1_tmp, inte_logit_tmp2_tmp])
                # inte_logit_tmp3 = tf.reduce_mean(inte_logit_tmp2, axis=2, keep_dims=False)
                # inte_logit = tf.reduce_mean(inte_logit_tmp3, axis=0, keep_dims=False)
                # gradients = tf.gradients(inte_logit, [interpolated, ])
                gradients = tf.gradients(inte_logit, [interpolated, ])[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(
                    tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)

                self.d_loss_patchGAN += self.config.lambda_gp * gradient_penalty
                self.g_loss_patchGAN = tf.reduce_mean(predict_fake_logit * -1)
            elif self.config.patchGAN_loss == "wgan":
                self.d_loss_patchGAN = tf.reduce_mean(
                    predict_fake_logit - predict_real_logit)
                self.g_loss_patchGAN = tf.reduce_mean(predict_fake_logit * -1)
            else:
                EPS = 1e-12
                self.d_loss_patchGAN = tf.reduce_mean(
                    -(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
                self.g_loss_patchGAN = tf.reduce_mean(
                    -tf.log(predict_fake + EPS))
        else:
            self.d_loss_patchGAN = 0
            self.g_loss_patchGAN = 0

        if self.config.type is "dcgan":
            if self.config.use_D_origin:
                self.d_loss_real = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.D_logits, tf.ones_like(self.D)))
                self.d_loss_fake = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.D_logits_, tf.zeros_like(self.D_)))
                self.d_loss = self.d_loss_real + self.d_loss_fake
                self.g_loss = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.D_logits_, tf.ones_like(self.D_)))
            else:
                self.d_loss = 0
                self.g_loss = 0

            # add patch discrimiative loss
            if self.config.use_D_patch:
                self.d_loss_real_p = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch_D_logits, tf.ones_like(self.patch_D)))
                self.d_loss_fake_p = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch_D_logits_, tf.zeros_like(self.patch_D_)))
                self.d_loss_patch = self.d_loss_real_p + self.d_loss_fake_p
                self.g_loss_patch = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch_D_logits_, tf.ones_like(self.patch_D_)))
            else:
                self.d_loss_patch = 0
                self.g_loss_patch = 0

            # add patch2 discrimiative loss
            if self.config.use_D_patch2:
                self.d_loss_real_p2 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch2_D_logits, tf.ones_like(self.patch2_D)))
                self.d_loss_fake_p2 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch2_D_logits_, tf.zeros_like(self.patch2_D_)))
                self.d_loss_patch2 = self.d_loss_real_p2 + self.d_loss_fake_p2
                self.g_loss_patch2 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch2_D_logits_, tf.ones_like(self.patch2_D_)))
            else:
                self.d_loss_patch2 = 0
                self.g_loss_patch2 = 0

            if self.config.use_D_patch2_2:
                self.d_loss_real_p2_2 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch2_2_D_logits, tf.ones_like(self.patch2_2_D)))
                self.d_loss_fake_p2_2 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch2_2_D_logits_, tf.zeros_like(self.patch2_2_D_)))
                self.d_loss_patch2_2 = self.d_loss_real_p2_2 + self.d_loss_fake_p2_2
                self.g_loss_patch2_2 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch2_2_D_logits_, tf.ones_like(self.patch2_2_D_)))
            else:
                self.d_loss_patch2_2 = 0
                self.g_loss_patch2_2 = 0

            # add patch3 discrimiative loss
            if self.config.use_D_patch3:
                self.d_loss_real_p3 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch3_D_logits, tf.ones_like(self.patch3_D)))
                self.d_loss_fake_p3 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch3_D_logits_, tf.zeros_like(self.patch3_D_)))
                self.d_loss_patch3 = self.d_loss_real_p3 + self.d_loss_fake_p3
                self.g_loss_patch3 = tf.reduce_mean(
                    networks.sigmoid_cross_entropy_with_logits(
                        self.patch3_D_logits_, tf.ones_like(self.patch3_D_)))
            else:
                self.d_loss_patch3 = 0
                self.g_loss_patch3 = 0

            # if self.config.G_num == 2:
            #     self.g1_loss = self.g_loss + self.config.D_patch3_loss * self.g_loss_patch3
            #     self.g2_loss = self.config.D_origin_loss * self.g_loss + self.config.D_patch_loss * self.g_loss_patch \
            #               + self.config.D_patch2_loss * self.g_loss_patch2 + self.d_loss_patchGAN * self.config.D_patchGAN_loss
            # else:
            #     self.g_loss = self.config.D_origin_loss * self.g_loss + self.config.D_patch_loss * self.g_loss_patch \
            #               + self.config.D_patch2_loss * self.g_loss_patch2 + self.config.D_patch3_loss * self.g_loss_patch3

        elif self.config.type is "wgan":
            self.d_loss_real = 0.0
            self.d_loss_fake = 0.0

            if self.config.use_D_origin:
                self.d_loss = tf.reduce_mean(self.D_logits_ - self.D_logits)
                self.g_loss = tf.reduce_mean(self.D_logits_*-1)
            else:
                self.d_loss = 0
                self.g_loss = 0

            # add patch discrimiative loss
            if self.config.use_D_patch:
                self.d_loss_patch = tf.reduce_mean(
                    self.patch_D_logits_ - self.patch_D_logits)
                self.g_loss_patch = tf.reduce_mean(self.patch_D_logits_ * -1)
            else:
                self.d_loss_patch = 0
                self.g_loss_patch = 0

            # add patch2 discrimiative loss
            if self.config.use_D_patch2:
                self.d_loss_patch2 = tf.reduce_mean(
                    self.patch2_D_logits_ - self.patch2_D_logits)
                self.g_loss_patch2 = tf.reduce_mean(self.patch2_D_logits_ * -1)
            else:
                self.d_loss_patch2 = 0
                self.g_loss_patch2 = 0

            if self.config.use_D_patch2_2:
                self.d_loss_patch2_2 = tf.reduce_mean(
                    self.patch2_2_D_logits_ - self.patch2_2_D_logits)
                self.g_loss_patch2_2 = tf.reduce_mean(
                    self.patch2_2_D_logits_ * -1)
            else:
                self.d_loss_patch2_2 = 0
                self.g_loss_patch2_2 = 0

            # add patch3 discrimiative loss
            if self.config.use_D_patch3:
                self.d_loss_patch3 = tf.reduce_mean(
                    self.patch3_D_logits_ - self.patch3_D_logits)
                self.g_loss_patch3 = tf.reduce_mean(self.patch3_D_logits_ * -1)
            else:
                self.d_loss_patch3 = 0
                self.g_loss_patch3 = 0

            # if self.config.G_num == 2:
            #     self.g1_loss = self.g_loss
            #     self.g2_loss = self.config.D_origin_loss * self.g_loss + self.config.D_patch_loss * self.g_loss_patch \
            #               + self.config.D_patch2_loss * self.g_loss_patch2 + self.config.D_patch3_loss * self.g_loss_patch3
            # else:
            #     self.g_loss = self.config.D_origin_loss * self.g_loss + self.config.D_patch_loss * self.g_loss_patch \
            #               + self.config.D_patch2_loss * self.g_loss_patch2 + self.config.D_patch3_loss * self.g_loss_patch3

        elif self.config.type is "gpwgan":
            self.d_loss_real = 0.0
            self.d_loss_fake = 0.0

            if self.config.use_D_origin:
                os.system('rm checksum/interpolated_origin')
                os.system('rm checksum/inte_logit_origin')
                os.system('rm checksum/d_loss')
                os.system('rm checksum/d_loss_gp')
                os.system('rm checksum/gradients_origin')
                os.system('rm checksum/grad_l2_origin')
                os.system('rm checksum/grad_result_origin')
                self.d_loss = tf.reduce_mean(self.D_logits_ - self.D_logits)

                alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
                alpha = alpha_dist.sample((self.config.batch_size, 1, 1, 1))
                if self.config.G_num == 2:
                    if self.config.originD_inputForm == "concat_w":
                        interpolated = self.inputs + \
                            alpha*(self.G_all-self.inputs)
                    elif self.config.originD_inputForm == "concat_n":
                        interpolated = self.concat_in_n + alpha * \
                            (self.concat_out_n - self.concat_in_n)
                else:
                    if self.config.originD_inputForm == "concat_w":
                        interpolated = self.inputs + alpha*(self.G-self.inputs)
                    elif self.config.originD_inputForm == "concat_n":
                        interpolated = self.concat_in_n + alpha * \
                            (self.concat_out_n - self.concat_in_n)
                interpolated = tf.keras.layers.Lambda(
                    save_tensor('interpolated_origin'))(interpolated)
                inte_logit = self.discriminator(interpolated, reuse=True)
                inte_logit = tf.keras.layers.Lambda(
                    save_tensor('inte_logit_origin'))(inte_logit)
                gradients = tf.gradients(inte_logit, [interpolated, ])[0]
                gradients = tf.keras.layers.Lambda(
                    save_tensor('gradients_origin'))(gradients)
                grad_l2 = tf.sqrt(tf.reduce_sum(
                    tf.square(gradients), axis=[1, 2, 3]))
                grad_l2 = tf.keras.layers.Lambda(
                    save_tensor('grad_l2_origin'))(grad_l2)
                gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
                gradient_penalty = tf.keras.layers.Lambda(
                    save_tensor('grad_result_origin'))(gradient_penalty)
                gradient_penalty = self.config.lambda_gp * gradient_penalty
                self.d_loss = tf.keras.layers.Lambda(
                    save_tensor('d_loss'))(self.d_loss)
                gradient_penalty = tf.keras.layers.Lambda(
                    save_tensor('d_loss_gp'))(gradient_penalty)

                self.d_loss += gradient_penalty
                self.g_loss = tf.reduce_mean(self.D_logits_ * -1)
            else:
                self.d_loss = 0
                self.g_loss = 0

            if self.config.use_D_patch:
                self.d_loss_patch = tf.reduce_mean(
                    self.patch_D_logits_ - self.patch_D_logits)

                alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
                alpha = alpha_dist.sample((self.config.batch_size, 1, 1, 1))
                interpolated = self.inputs + alpha * (self.G - self.inputs)
                inte_logit = self.discriminator_patch(interpolated, reuse=True)
                gradients = tf.gradients(inte_logit, [interpolated, ])[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(
                    tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)

                self.d_loss_patch += self.config.lambda_gp * gradient_penalty

                self.g_loss_patch = tf.reduce_mean(self.patch_D_logits_ * -1)
            else:
                self.d_loss_patch = 0
                self.g_loss_patch = 0

            if self.config.use_D_patch2:
                self.patch2_D_logits_ = save_layer('patch2_D_logits__origin', self.patch2_D_logits_)
                self.patch2_D_logits = save_layer('patch2_D_logits_origin', self.patch2_D_logits)
                self.d_loss_patch2 = tf.reduce_mean(
                    self.patch2_D_logits_ - self.patch2_D_logits)
                self.resized_G2_p2 = save_layer(
                    'resized_G2_p2_origin', self.resized_G2_p2)
                alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
                alpha = alpha_dist.sample((self.config.batch_size, 1, 1, 1))

                # TODO: Not know whtether it's true

                interpolated = self.resized_inputs + alpha * \
                    (self.resized_G2_p2 - self.resized_inputs)

                inte_logit = self.discriminator_patch2(
                    interpolated, reuse=True)
                gradients = tf.gradients(inte_logit, [interpolated, ])[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(
                    tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)
                grad_pen = self.config.lambda_gp * gradient_penalty
                self.d_loss_patch2 = save_layer(
                    'image_d_loss_origin', self.d_loss_patch2)
                grad_pen = save_layer(
                    'image_d_loss_grad_penalty_origin', grad_pen)

                self.d_loss_patch2 += grad_pen
                self.d_loss_patch2 = save_layer(
                    'd_loss_patch2_origin', self.d_loss_patch2)
                self.g_loss_patch2 = tf.reduce_mean(self.patch2_D_logits_ * -1)
            else:
                self.d_loss_patch2 = 0
                self.g_loss_patch2 = 0

            if self.config.use_D_patch2_2:
                self.d_loss_patch2_2 = tf.reduce_mean(
                    self.patch2_2_D_logits_ - self.patch2_2_D_logits)

                alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
                alpha = alpha_dist.sample((self.config.batch_size, 1, 1, 1))

                # TODO: Not know whtether it's true
                interpolated = self.resized_inputs_p2_2 + alpha * \
                    (self.resized_G2_p2_2 - self.resized_inputs_p2_2)

                inte_logit = self.discriminator_patch2_2(
                    interpolated, reuse=True)
                gradients = tf.gradients(inte_logit, [interpolated, ])[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(
                    tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)

                self.d_loss_patch2_2 += self.config.lambda_gp * gradient_penalty
                self.g_loss_patch2_2 = tf.reduce_mean(
                    self.patch2_2_D_logits_ * -1)
            else:
                self.d_loss_patch2_2 = 0
                self.g_loss_patch2_2 = 0

            if self.config.use_D_patch3:
                self.d_loss_patch3 = tf.reduce_mean(
                    self.patch3_D_logits_ - self.patch3_D_logits)

                alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
                alpha = alpha_dist.sample((self.config.batch_size, 1, 1, 1))

                # TODO: Not know whtether it's true
                interpolated = self.resized_inputs_p3 + alpha * \
                    (self.resized_G1_p3 - self.resized_inputs_p3)

                inte_logit = self.discriminator_patch3(
                    interpolated, reuse=True)
                gradients = tf.gradients(inte_logit, [interpolated, ])[0]
                grad_l2 = tf.sqrt(tf.reduce_sum(
                    tf.square(gradients), axis=[1, 2, 3]))
                gradient_penalty = tf.reduce_mean((grad_l2 - 1) ** 2)

                self.d_loss_patch3 += self.config.lambda_gp * gradient_penalty
                self.d_loss_patch3 = save_layer(
                    'd_loss_patch3_origin', self.d_loss_patch3)
                self.g_loss_patch3 = tf.reduce_mean(self.patch3_D_logits_ * -1)
            else:
                self.d_loss_patch3 = 0
                self.g_loss_patch3 = 0

        if self.config.G_num == 2:
            self.g1_loss = self.config.D_origin_loss * self.g_loss + self.config.D_patch3_loss * \
                self.g_loss_patch3 + self.d_loss_patchGAN * self.config.D_patchGAN_loss
            self.g2_loss = self.config.D_origin_loss * self.g_loss + self.config.D_patch_loss * self.g_loss_patch \
                + self.config.D_patch2_loss * self.g_loss_patch2 + self.d_loss_patchGAN * \
                self.config.D_patchGAN_loss + self.config.D_patch2_2_loss * self.g_loss_patch2_2
        else:
            self.g_loss = self.config.D_origin_loss * self.g_loss + self.config.D_patch_loss * self.g_loss_patch \
                + self.config.D_patch2_loss * self.g_loss_patch2 + \
                self.config.D_patch3_loss * self.g_loss_patch3

        # focal loss
        if self.config.if_focal_loss:
            self.loss_g_ac, self.loss_d_ac = networks.get_acgan_loss_focal(
                self.D_logits2, tf.cast(self.z[:, -1], dtype=tf.int32),
                self.D_logits2_, tf.cast(self.z[:, -1], dtype=tf.int32),
                num_classes=self.config.num_classes)

            if self.config.G_num == 2:
                self.g2_loss += self.loss_g_ac
            else:
                self.g_loss += self.loss_g_ac
            # self.d_loss += self.loss_d_ac
        else:
            self.loss_g_ac = 0
            self.loss_d_ac = 0

        if self.config.E_stage1:
            if self.config.if_focal_loss:
                self.zl_loss = tf.reduce_mean(
                    tf.abs(self.z[:, 0:self.z_dim] - z_recon)) * self.config.stage1_zl_loss

                # l1
                # class_recon = tf.nn.softmax(class_recon)
                # self.class_loss = tf.reduce_mean(tf.abs(self.class_onehot - class_recon))

                # focal loss
                if self.config.E2_stage1:
                    self.class_loss = networks.get_class_loss(class_recon, tf.cast(
                        self.z[:, -1], dtype=tf.int32), self.config.num_classes)
                else:
                    self.class_loss = 0
            else:
                self.zl_loss = tf.reduce_mean(
                    tf.abs(self.z - z_recon)) * self.config.stage1_zl_loss
                self.class_loss = 0
        else:
            self.zl_loss = 0
            self.class_loss = 0

        # self.g_loss = self.g_loss + self.zl_loss * self.config.stage1_zl_loss

        # optimizer
        if self.config.use_patchGAN_D_full == True and self.config.G_num == 2:
            self.d_optim_patchGAN = tf.train.AdamOptimizer(self.config.learning_rate,
                                                           beta1=self.config.beta1).minimize(
                self.d_loss_patchGAN, var_list=self.discriminator_patchGAN.var_list)

        if self.config.optim is "adam":
            if self.config.use_D_origin:
                self.d_optim = tf.train.AdamOptimizer(self.config.learning_rate,
                                                      beta1=self.config.beta1).minimize(
                    self.d_loss, var_list=self.discriminator.var_list)

            if self.config.use_D_patch:
                self.d_optim_patch = tf.train.AdamOptimizer(self.config.learning_rate,
                                                            beta1=self.config.beta1).minimize(
                    self.d_loss_patch, var_list=self.discriminator_patch.var_list)
            if self.config.use_D_patch2:
                self.d_optim_patch2 = tf.train.AdamOptimizer(self.config.learning_rate,
                                                             beta1=self.config.beta1).minimize(
                    self.d_loss_patch2, var_list=self.discriminator_patch2.var_list)
            if self.config.use_D_patch2_2:
                self.d_optim_patch2_2 = tf.train.AdamOptimizer(self.config.learning_rate,
                                                               beta1=self.config.beta1).minimize(
                    self.d_loss_patch2_2, var_list=self.discriminator_patch2_2.var_list)
            if self.config.use_D_patch3:
                self.d_optim_patch3 = tf.train.AdamOptimizer(self.config.learning_rate,
                                                             beta1=self.config.beta1).minimize(
                    self.d_loss_patch3, var_list=self.discriminator_patch3.var_list)
            if self.config.if_focal_loss:
                self.d_optim2 = tf.train.AdamOptimizer(self.config.learning_rate,
                                                       beta1=self.config.beta1).minimize(
                    self.loss_d_ac, var_list=self.discriminator2.var_list)

            if self.config.G_num == 2:
                self.g1_optim = tf.train.AdamOptimizer(self.config.learning_rate,
                                                       beta1=self.config.beta1).minimize(
                    self.g1_loss, var_list=self.generator1.var_list)
                self.g2_optim = tf.train.AdamOptimizer(self.config.learning_rate,
                                                       beta1=self.config.beta1).minimize(
                    self.g2_loss, var_list=self.generator2.var_list)
            else:
                self.g_optim = tf.train.AdamOptimizer(self.config.learning_rate,
                                                      beta1=self.config.beta1).minimize(
                    self.g_loss, var_list=self.generator.var_list)

            if self.config.E_stage1:
                self.e_optim = tf.train.AdamOptimizer(self.config.learning_rate,
                                                      beta1=self.config.beta1).minimize(
                    self.zl_loss, var_list=self.encoder.var_list)
                if self.config.if_focal_loss and self.config.E2_stage1:
                    self.e_optim2 = tf.train.AdamOptimizer(self.config.learning_rate,
                                                           beta1=self.config.beta1).minimize(
                        self.class_loss, var_list=self.encoder2.var_list)
        elif self.config.optim == "rmsprop":
            pass
            # if self.config.use_D_origin:
            #     self.d_optim = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.d_loss, var_list=self.discriminator.var_list)
            # if self.config.use_D_patch:
            #     self.d_optim_patch = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.d_loss_patch, var_list=self.discriminator_patch.var_list)
            # if self.config.use_D_patch2:
            #     self.d_optim_patch2 = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.d_loss_patch2, var_list=self.discriminator_patch2.var_list)
            # if self.config.use_D_patch2_2:
            #     self.d_optim_patch2_2 = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.d_loss_patch2_2, var_list=self.discriminator_patch2_2.var_list)

            # if self.config.use_D_patch3:
            #     self.d_optim_patch3 = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.d_loss_patch3, var_list=self.discriminator_patch3.var_list)
            # if self.config.if_focal_loss:
            #     self.d_optim2 = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.loss_d_ac, var_list=self.discriminator2.var_list)

            # if self.config.G_num == 2:
            #     self.g1_optim = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.g1_loss, var_list=self.generator1.var_list)
            #     self.g2_optim = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.g2_loss, var_list=self.generator2.var_list)
            # else:
            #     self.g_optim = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.g_loss, var_list=self.generator.var_list)

            # if self.config.E_stage1:
            #     self.e_optim = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #         self.zl_loss, var_list=self.encoder.var_list)
            #     if self.config.if_focal_loss and self.config.E2_stage1:
            #         self.e_optim2 = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            #             self.class_loss, var_list=self.encoder2.var_list)
        # ??? something not understood
        if self.config.type is "wgan":
            if self.config.use_D_origin:
                clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.config.clamp_lower, self.config.clamp_upper))
                                 for var in self.discriminator.var_list]
                # merge the clip operations on critic variables
                with tf.control_dependencies([self.d_optim]):
                    self.d_optim = tf.tuple(clipped_var_c)

            if self.config.use_D_patch:
                clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.config.clamp_lower, self.config.clamp_upper))
                                 for var in self.discriminator_patch.var_list]
                # merge the clip operations on critic variables
                with tf.control_dependencies([self.d_optim_patch]):
                    self.d_optim_patch = tf.tuple(clipped_var_c)

            if self.config.use_D_patch2:
                clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.config.clamp_lower, self.config.clamp_upper))
                                 for var in self.discriminator_patch2.var_list]
                # merge the clip operations on critic variables
                with tf.control_dependencies([self.d_optim_patch2]):
                    self.d_optim_patch2 = tf.tuple(clipped_var_c)

            if self.config.use_D_patch2_2:
                clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.config.clamp_lower, self.config.clamp_upper))
                                 for var in self.discriminator_patch2_2.var_list]
                # merge the clip operations on critic variables
                with tf.control_dependencies([self.d_optim_patch2_2]):
                    self.d_optim_patch2_2 = tf.tuple(clipped_var_c)

            if self.config.use_D_patch3:
                clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.config.clamp_lower, self.config.clamp_upper))
                                 for var in self.discriminator_patch3.var_list]
                # merge the clip operations on critic variables
                with tf.control_dependencies([self.d_optim_patch3]):
                    self.d_optim_patch3 = tf.tuple(clipped_var_c)

            if self.config.G_num == 2 and self.config.use_patchGAN_D_full == True and self.config.patchGAN_loss == "wgan":
                clipped_var_c = [tf.assign(var, tf.clip_by_value(var, self.config.clamp_lower, self.config.clamp_upper))
                                 for var in self.discriminator_patchGAN.var_list]
                # merge the clip operations on critic variables
                with tf.control_dependencies([self.d_optim_patchGAN]):
                    self.d_optim_patchGAN = tf.tuple(clipped_var_c)

        # define sumary
        # self.z_sum = networks.histogram_summary("z", self.z)
        # self.inputs_sum = networks.image_summary("inputs", self.inputs)

        # if self.config.G_num == 2:
        #     self.G1_sum = networks.image_summary("G1", self.G1)
        #     self.G2_sum = networks.image_summary("G2", self.G2)
        # else:
        #     self.G_sum = networks.image_summary("G", self.G)

        # self.d_loss_real_sum = networks.scalar_summary(
        #     "d_loss_real", self.d_loss_real)
        # self.d_loss_fake_sum = networks.scalar_summary(
        #     "d_loss_fake", self.d_loss_fake)

        # if self.config.G_num == 2:
        #     self.g1_loss_sum = networks.scalar_summary("g1_loss", self.g1_loss)
        #     self.g2_loss_sum = networks.scalar_summary("g2_loss", self.g2_loss)

        # self.g_loss_sum = networks.scalar_summary("g_loss", self.g_loss)

        # self.d_loss_sum = networks.scalar_summary("d_loss", self.d_loss)

        # self.zl_loss_sum = networks.scalar_summary("zl_loss", self.zl_loss)

        # self.class_loss_sum = networks.scalar_summary(
        #     "class_loss", self.class_loss)

        # self.loss_g_ac_sum = networks.scalar_summary(
        #     "loss_g_ac", self.loss_g_ac)
        # self.loss_d_ac_sum = networks.scalar_summary(
        #     "loss_d_ac", self.loss_d_ac)

        # if self.config.G_num == 2:
        #     self.g_sum = networks.merge_summary([self.z_sum, self.G1_sum, self.G2_sum,
        #                                          self.d_loss_fake_sum, self.zl_loss_sum, self.g_loss_sum,
        #                                          self.class_loss_sum, self.loss_g_ac_sum, self.g1_loss_sum, self.g2_loss_sum])
        #     if self.config.use_patchGAN_D_full:
        #         self.d_loss_patchGAN_sum = networks.scalar_summary(
        #             "d_loss_patchGAN", self.d_loss_patchGAN)
        #         self.g_sum = networks.merge_summary(
        #             [self.g_sum, self.d_loss_patchGAN_sum])
        # else:
        #     self.g_sum = networks.merge_summary([self.z_sum, self.G_sum,
        #                                          self.d_loss_fake_sum, self.g_loss_sum, self.zl_loss_sum,
        #                                          self.class_loss_sum, self.loss_g_ac_sum])
        # self.d_sum = networks.merge_summary([self.z_sum, self.inputs_sum,
        #                                      self.d_loss_real_sum, self.d_loss_sum, self.loss_d_ac_sum])
        # if self.config.use_D_origin:
        #     self.d_sum_tmp = networks.histogram_summary("d", self.D)
        #     self.d__sum_tmp = networks.histogram_summary("d_", self.D_)
        #     self.g_sum = networks.merge_summary([self.g_sum, self.d__sum_tmp])
        #     self.d_sum = networks.merge_summary([self.d_sum, self.d_sum_tmp])

        # if self.config.use_D_patch:
        #     self.d_patch_sum = networks.histogram_summary(
        #         "patch_d", self.patch_D)
        #     self.d__patch_sum = networks.histogram_summary(
        #         "patch_d_", self.patch_D_)
        #     self.d_loss_patch_sum = networks.scalar_summary(
        #         "d_loss_patch", self.d_loss_patch)
        #     self.g_loss_patch_sum = networks.scalar_summary(
        #         "g_loss_patch", self.g_loss_patch)
        #     self.g_sum = networks.merge_summary(
        #         [self.g_sum, self.d__patch_sum, self.g_loss_patch_sum])
        #     self.d_sum = networks.merge_summary(
        #         [self.d_sum, self.d_patch_sum, self.d_loss_patch_sum])

        # if self.config.use_D_patch2:
        #     self.d_patch2_sum = networks.histogram_summary(
        #         "patch2_d", self.patch2_D)
        #     self.d__patch2_sum = networks.histogram_summary(
        #         "patch2_d_", self.patch2_D_)
        #     self.resized_inputs_sum = networks.image_summary(
        #         "resized_inputs_image", self.resized_inputs_image)
        #     if self.config.G_num == 2:
        #         self.resized_G_sum = networks.image_summary(
        #             "resized_G_image", self.resized_G2_p2)
        #     else:
        #         self.resized_G_sum = networks.image_summary(
        #             "resized_G_image", self.resized_G_image)
        #     self.d_loss_patch2_sum = networks.scalar_summary(
        #         "d_loss_patch2", self.d_loss_patch2)
        #     self.g_loss_patch2_sum = networks.scalar_summary(
        #         "g_loss_patch2", self.g_loss_patch2)
        #     self.g_sum = networks.merge_summary(
        #         [self.g_sum, self.d__patch2_sum, self.resized_G_sum, self.g_loss_patch2_sum])
        #     self.d_sum = networks.merge_summary(
        #         [self.d_sum, self.d_patch2_sum, self.resized_inputs_sum, self.d_loss_patch2_sum])

        # if self.config.use_D_patch3:
        #     self.d_patch3_sum = networks.histogram_summary(
        #         "patch3_d", self.patch3_D)
        #     self.d__patch3_sum = networks.histogram_summary(
        #         "patch3_d_", self.patch3_D_)
        #     self.resized_inputs_p3_sum = networks.image_summary(
        #         "resized_inputs_p3_image", self.resized_inputs_p3_image)
        #     if self.config.G_num == 2:
        #         self.resized_G_p3_sum = networks.image_summary(
        #             "resized_G_p3_image", self.resized_G1_p3)
        #     else:
        #         self.resized_G_p3_sum = networks.image_summary(
        #             "resized_G_p3_image", self.resized_G_p3_image)
        #     self.d_loss_patch3_sum = networks.scalar_summary(
        #         "d_loss_patch3", self.d_loss_patch3)
        #     self.g_loss_patch3_sum = networks.scalar_summary(
        #         "g_loss_patch3", self.g_loss_patch3)
        #     self.g_sum = networks.merge_summary(
        #         [self.g_sum, self.d__patch3_sum, self.resized_G_p3_sum, self.g_loss_patch3_sum])
        #     self.d_sum = networks.merge_summary(
        #         [self.d_sum, self.d_patch3_sum, self.resized_inputs_p3_sum, self.d_loss_patch3_sum])

        if self.config.G_num == 2:
            self.saver = tf.train.Saver(
                {v.op.name: v for v in self.generator1.var_list + self.generator2.var_list})
        else:
            self.saver = tf.train.Saver(
                {v.op.name: v for v in self.generator.var_list + self.discriminator.var_list})
        self.saver2 = tf.train.Saver()

        utils.show_all_variables()

    def train1(self):
        def extension(filename):
            return os.path.splitext(filename)[-1]

        def checksum_save(input_dict):
            checksum_path = utils.checksum_path
            utils.makedirs(checksum_path)

            def save(filename, obj):
                p = os.path.join(checksum_path, filename)
                if isinstance(obj, np.ndarray):
                    np.save(p + '.npy', val)
                else:
                    with open(p + '.pkl', 'wb') as f:
                        pickle.dump(obj, f)

            for key, val in input_dict.items():
                save(key, val)

        def checksum_load(*names):
            def load(filename):
                if extension(path) == '.npy':
                    return np.load(path)
                elif extension(path) == '.pkl':
                    with open(path, 'rb') as f:
                        return pickle.load(f)
                else:
                    raise NotImplementedError

            def enforce_exists(path):
                if not os.path.exists(path):
                    print('missing loading item: {}'.format(path))
                    raise ValueError

            checksum_path = utils.checksum_path
            result = []
            for name in names:
                path = os.path.join(checksum_path, name)
                enforce_exists(path)
                result.append(load(path))
            return result

        self.build_model1()

        # load data
        if self.config.if_focal_loss:
            self.data = []
            for i in xrange(self.config.num_classes):
                data_path = os.path.join(self.config.data_dir,
                                         self.config.dataset,
                                         "stage1", "train", str(i),
                                         "*.png")
                self.data.extend(glob(data_path))
                data_path = os.path.join(self.config.data_dir,
                                         self.config.dataset,
                                         "stage1", "train", str(i),
                                         "*.jpg")
                self.data.extend(glob(data_path))
        else:
            data_path = os.path.join(self.config.data_dir,
                                     self.config.dataset,
                                     "stage1", "train",
                                     self.config.input_fname_pattern)
            self.data = glob(data_path)

        if len(self.data) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")
        if len(self.data) < self.config.batch_size:
            raise Exception(
                "[!] Entire dataset size is less than the configured batch_size")

        self.grayscale = (self.c_dim == 1)

        # init var
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # init summary writer
        self.writer = networks.SummaryWriter("./logs", self.sess.graph)
        # self.writer = networks.SummaryWriter("./logs_2", self.sess.graph)

        # load model if exist
        counter = 1
        start_time = time.time()
        # if self.config.E_stage1:
        #     could_load, checkpoint_counter = self.load(
        #         self.saver2, self.config.checkpoint_dir+"/stage1_AddE", self.model_dir)
        # else:
        #     could_load, checkpoint_counter = self.load(self.saver2, self.config.checkpoint_dir + "/stage1",
        #                                                self.model_dir)
        # if could_load:
        #     counter = checkpoint_counter
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")

        # train
        for epoch in xrange(self.config.epoch):
            np.random.shuffle(self.data)
            batch_idxs = min(
                len(self.data), self.config.train_size) // self.config.batch_size

            for idx in xrange(batch_idxs):
                # read image by batch
                batch_files = self.data[idx *
                                        self.config.batch_size: (idx+1)*self.config.batch_size]
                batch = [
                    utils.get_image(batch_file,
                                    input_height=self.config.input_height,
                                    input_width=self.config.input_width,
                                    resize_height=self.config.output_height,
                                    resize_width=self.config.output_width,
                                    crop=self.config.crop,
                                    grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(
                        np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # init z by batch

                batch_z = np.random.normal(
                    size=(self.config.batch_size, self.z_dim))
                if idx == 3:
                    checksum_save({
                        'batch_files': batch_files,
                        'batch_images': batch_images,
                        'batch_z': batch_z,
                    })

                    # (restore_batch_files, restore_batch_images, restore_batch_z) = checksum_load(
                    #     'batch_files.pkl', 'batch_images.npy', 'batch_z.npy')
                    # assert np.all(batch_images == restore_batch_images)
                    # assert np.all(batch_z == restore_batch_z)
                    # for out, tar in zip(restore_batch_files, batch_files):
                    #     assert out == tar
                    # print('assertion successed!')

                if self.config.if_focal_loss:
                    def getClass(filePath):
                        end = filePath.rfind("/")
                        start = filePath.rfind("/", 0, end)
                        return int(filePath[start+1:end])
                    batch_classes = [getClass(batch_file)
                                     for batch_file in batch_files]
                    batch_classes = np.array(batch_classes).reshape(
                        (self.config.batch_size, 1))
                    batch_z = np.concatenate((batch_z, batch_classes), axis=1)

                # if self.config.E_stage1:
                #     self.save(self.saver2, self.config.checkpoint_dir +
                #               "/stage1_AddE", self.model_dir, counter)
                # else:
                #     self.save(self.saver, self.config.checkpoint_dir +
                #               "/stage1", self.model_dir, counter)

                # # Update D network
                # if self.config.use_D_origin:
                #     _ = self.sess.run([self.d_optim],
                #                       feed_dict={self.inputs: batch_images, self.z: batch_z})

                # if self.config.use_D_patch:
                #     _ = self.sess.run([self.d_optim_patch],
                #                       feed_dict={self.inputs: batch_images, self.z: batch_z})

                # if self.config.use_D_patch2:
                #     _ = self.sess.run([self.d_optim_patch2],
                #                       feed_dict={self.inputs: batch_images, self.z: batch_z})

                # if self.config.use_D_patch2_2:
                #     _ = self.sess.run([self.d_optim_patch2_2],
                #                       feed_dict={self.inputs: batch_images, self.z: batch_z})
                # if self.config.use_D_patch3:
                #     _ = self.sess.run([self.d_optim_patch3],
                #                       feed_dict={self.inputs: batch_images, self.z: batch_z})

                # if self.config.use_patchGAN_D_full == True and self.config.G_num == 2:
                #     _ = self.sess.run([self.d_optim_patchGAN],
                #                       feed_dict={self.inputs: batch_images, self.z: batch_z})

                # if self.config.if_focal_loss:
                #     _ = self.sess.run(self.d_optim2,
                #                       feed_dict={self.inputs: batch_images, self.z: batch_z})

                # if self.config.G_num == 2:
                #     _, _, summary_str = self.sess.run([self.g1_optim, self.g2_optim, self.g_sum],
                #                                       feed_dict={self.z: batch_z, self.inputs: batch_images})
                # else:
                #     _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                #                                    feed_dict={self.z: batch_z})

                # if self.config.E_stage1:
                #     _ = self.sess.run([self.e_optim],
                #                       feed_dict={self.z: batch_z})

                #     if self.config.if_focal_loss and self.config.E2_stage1:
                #         _ = self.sess.run([self.e_optim2],
                #                           feed_dict={self.inputs: batch_images, self.z: batch_z})

                # if self.config.G_num == 2:
                #     _, _, summary_str = self.sess.run([self.g1_optim, self.g2_optim, self.g_sum],
                #                                       feed_dict={self.z: batch_z, self.inputs: batch_images})
                # else:
                #     _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                #                                    feed_dict={self.z: batch_z})

                if self.config.type is "dcgan":
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval(
                        {self.inputs: batch_images})
                    if self.config.G_num == 2:
                        errG = self.g1_loss.eval(
                            {self.z: batch_z}) + self.g2_loss.eval({self.z: batch_z})
                    else:
                        errG = self.g_loss.eval({self.z: batch_z})
                else:
                    if self.config.use_D_origin:
                        errD_fake_tmp1 = self.d_loss.eval(
                            {self.inputs: batch_images, self.z: batch_z})
                    else:
                        errD_fake_tmp1 = 0
                    if self.config.use_D_patch:
                        errD_fake_tmp2 = self.d_loss_patch.eval(
                            {self.inputs: batch_images, self.z: batch_z})
                    else:
                        errD_fake_tmp2 = 0
                    if self.config.use_D_patch2:
                        errD_fake_tmp3 = self.d_loss_patch2.eval(
                            {self.inputs: batch_images, self.z: batch_z})
                    else:
                        errD_fake_tmp3 = 0
                    if self.config.use_D_patch3:
                        errD_fake_tmp4 = self.d_loss_patch3.eval(
                            {self.inputs: batch_images, self.z: batch_z})
                    else:
                        errD_fake_tmp4 = 0
                    errD_fake = errD_fake_tmp1 + errD_fake_tmp2 + errD_fake_tmp3 + errD_fake_tmp4
                    errD_real = errD_fake
                    if self.config.G_num == 2:
                        errG = self.g1_loss.eval({self.z: batch_z, self.inputs: batch_images}) + self.g2_loss.eval(
                            {self.z: batch_z, self.inputs: batch_images})
                        if self.config.use_patchGAN_D_full == True:
                            errD_patchGAN = self.d_loss_patchGAN.eval(
                                {self.inputs: batch_images, self.z: batch_z})
                    else:
                        errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                if self.config.G_num == 2 and self.config.use_patchGAN_D_full == True:
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_loss_patchGAN: %.8f"
                          % (epoch, self.config.epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake + errD_real, errG, errD_patchGAN))
                else:
                    print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                          % (epoch, self.config.epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake+errD_real, errG))

                outputL = self.sess.run(self.G1,
                                        feed_dict={self.z: batch_z, self.inputs: batch_images})

                if idx == 3:
                    checksum_save({
                        "outputL": outputL,
                        "errD_fake": errD_fake,
                        "errD_real": errD_real,
                        "errG": errG,
                    })
                    # restore_outputL, restore_errD_fake, restore_errD_real, restore_errG = checksum_load(
                    #     "outputL.npy", "errD_fake.pkl", "errD_real.pkl", "errG.pkl",)
                    # assert np.all(restore_outputL == outputL)
                    # assert errD_fake == restore_errD_fake
                    # assert errD_real == restore_errD_real
                    # assert errG == restore_errG
                    # print('assert successed!')
                    exit()
                # if np.mod(counter, self.config.save_checkpoint_frequency) == 2:
                # if self.config.E_stage1:
                #     self.save(self.saver2, self.config.checkpoint_dir + "/stage1_AddE", self.model_dir, counter)
                # else:
                #     self.save(self.saver, self.config.checkpoint_dir+"/stage1", self.model_dir, counter)

    def test1(self, num):
        self.build_model1()

        # init var
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # load model if exist
        counter = 1
        start_time = time.time()
        if self.config.E_stage1:
            # TODO: Why here must use saver rather than saver2?
            could_load, checkpoint_counter = self.load(self.saver, self.config.checkpoint_dir + "/stage1_AddE",
                                                       self.model_dir)
            # could_load, checkpoint_counter = self.load(self.saver, self.config.checkpoint_dir + "/stage2",
            #                                            self.model_dir2)
        else:
            could_load, checkpoint_counter = self.load(
                self.saver2, self.config.checkpoint_dir+"/stage1", self.model_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        # test
        batch_idxs = min(num, self.config.train_size) // self.config.batch_size

        for idx in xrange(0, int(batch_idxs)):
            # init z by batch
            batch_z = np.random.uniform(-1, 1,
                                        [self.config.batch_size, self.z_dim]).astype(np.float32)
            if self.config.if_focal_loss:
                if self.config.Test_singleLabel:
                    # batch_classes = np.floor(np.random.uniform(0, 5,
                    #                                            [self.config.batch_size, 1]).astype(
                    #     np.float32) * self.config.num_classes)
                    # batch_classes = tf.fill([self.config.batch_size, 1], idx)
                    batch_classes = np.full(
                        (self.config.batch_size, 1), idx, dtype=np.float32)
                    # batch_classes = tf.fill(batch_classes.size(), idx)
                    # batch_classes = np.floor(np.random.uniform(1, 2,
                    #                                            [self.config.batch_size, 1]).astype(
                    #     np.float32) * 1)
                    batch_z = np.concatenate((batch_z, batch_classes), axis=1)
                else:
                    batch_classes = np.floor(np.random.uniform(0, 5,
                                                               [self.config.batch_size, 1]).astype(np.float32)*self.config.num_classes)
                    # batch_classes = np.floor(np.random.uniform(1, 2,
                    #                                            [self.config.batch_size, 1]).astype(
                    #     np.float32) * 1)
                    batch_z = np.concatenate((batch_z, batch_classes), axis=1)
            # batch_z = np.random.normal(size=(self.config.batch_size, self.z_dim))
            #
            # if self.config.if_focal_loss:
            #     batch_classes = tf.floor(np.random.uniform(0, 1,
            #             [self.config.batch_size, 1]).astype(np.float32)*self.config.num_classes)
            #     batch_z = tf.concat([batch_z, batch_classes], 1)

            # generate images
            results = self.sess.run(self.G, feed_dict={self.z: batch_z})

            image_frame_dim = int(math.ceil(self.config.batch_size**.5))
            if self.config.E_stage1:
                utils.save_images(results, [image_frame_dim, image_frame_dim],
                                  self.config.sample_dir + '/stage1_AddE_random/' + self.config.dataset + '/' +
                                  self.model_dir + '__test_%s.png' % idx)
            else:
                utils.save_images(results, [image_frame_dim, image_frame_dim],
                                  self.config.sample_dir+'/stage1/'+self.model_dir+'__test_%s.png' % idx)

            print("Test: [%4d/%4d]" % (idx, batch_idxs))

    def build_model2(self):
        # Define models
        # self.encoder = Encoder('E', is_train=True,
        #                 image_size=self.config.input_height,
        #                 latent_dim=self.z_dim,
        #                 use_resnet=True)

        # for testing
        if self.config.output_form is "batch":
            batch_size_mid = self.config.batch_size
        else:
            batch_size_mid = 1

        #  for testing
        if (not self.config.Test_allLabel) or (self.config.Test_singleLabel and self.config.test_label == 0):
            if self.config.if_focal_loss:
                self.encoder = Encoder('E', is_train=True,
                                       norm=self.config.E_norm,
                                       image_size=self.config.input_height,
                                       latent_dim=self.z_dim,
                                       use_resnet=self.config.if_resnet_e)
                if self.config.Test_encoderSketch:
                    self.encoder2 = Encoder('E2', is_train=True,
                                            norm=self.config.E_norm,
                                            image_size=self.config.input_height,
                                            latent_dim=self.config.num_classes,
                                            use_resnet=self.config.if_resnet_e)
                if self.config.Test_classifierSketch:
                    self.discriminator2 = Discriminator2(
                        'D2', self.config.SPECTRAL_NORM_UPDATE_OPS)
            else:
                # TODO latent_dim is corret?
                self.encoder = Encoder('E', is_train=True,
                                       norm=self.config.E_norm,
                                       image_size=self.config.input_height,
                                       latent_dim=self.z_dim,
                                       use_resnet=self.config.if_resnet_e)

            if self.config.model is "old":
                if self.config.G_num == 2:
                    self.generator1 = Generator('G1', is_train=False,
                                                norm=self.config.G_norm,
                                                batch_size=batch_size_mid,
                                                output_height=self.config.output_height,
                                                output_width=int(
                                                    self.config.output_width/2),
                                                input_dim=self.gf_dim,
                                                output_dim=self.c_dim,
                                                use_resnet=self.config.if_resnet_g)
                    self.generator2 = Generator('G2', is_train=False,
                                                norm=self.config.G_norm,
                                                batch_size=batch_size_mid,
                                                output_height=self.config.output_height,
                                                output_width=int(
                                                    self.config.output_width/2),
                                                input_dim=self.gf_dim,
                                                output_dim=self.c_dim,
                                                use_resnet=self.config.if_resnet_g)
                else:
                    self.generator = Generator('G', is_train=False,
                                               norm=self.config.G_norm,
                                               batch_size=batch_size_mid,
                                               output_height=self.config.output_height,
                                               output_width=self.config.output_width,
                                               input_dim=self.gf_dim,
                                               output_dim=self.c_dim,
                                               use_resnet=self.config.if_resnet_g)
                # self.discriminator = Discriminator('D', is_train=False,
                #                                 norm=self.config.D_norm,
                #                                 num_filters=self.df_dim,
                #                                    use_resnet=self.config.if_resnet_d)
            else:
                self.generator = base_model.generator_model
                # self.discriminator = base_model.discriminator_model

        # define inputs
        if self.config.crop:
            self.image_dims = [self.config.output_height, self.config.output_width,
                               self.c_dim]
        else:
            self.image_dims = [self.config.input_height, self.config.input_width,
                               self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [batch_size_mid] + self.image_dims, name='real_images')

        self.masks = tf.placeholder(
            tf.float32, [batch_size_mid] + self.image_dims, name='mask_images')

        self.input_left = self.inputs[0:batch_size_mid, 0:self.image_dims[0], 0:int(
            self.image_dims[1]/2), 0:self.image_dims[2]]
        z_encoded, z_encoded_mu, z_encoded_log_sigma = self.encoder(
            self.input_left)
        # matrix_min = tf.reduce_min(z_encoded)
        # matrix_max = tf.reduce_max(z_encoded)
        # self.z = (z_encoded - matrix_min)*2/(matrix_max-matrix_min) -1
        self.z = z_encoded
        if self.config.if_focal_loss:
            if self.config.Test_singleLabel:
                batch_classes = np.full(
                    (batch_size_mid, 1), self.config.test_label, dtype=np.float32)
                # batch_z = np.concatenate((z_encoded, batch_classes), axis=1)
                self.class_onehot = tf.one_hot(tf.cast(batch_classes[:, -1], dtype=tf.int32), self.config.num_classes,
                                               on_value=1., off_value=0., dtype=tf.float32)
                self.z = tf.concat([z_encoded, self.class_onehot], 1)
            elif self.config.Test_encoderSketch:
                class_encoded, _, _ = self.encoder2(self.input_left)
                self.class_onehot = tf.one_hot(tf.cast(tf.argmax(class_encoded, 1), dtype=tf.int32), self.config.num_classes,
                                               on_value=1., off_value=0., dtype=tf.float32)
                self.z = tf.concat([z_encoded, self.class_onehot], 1)
            elif self.config.Test_classifierSketch:
                self.input_right = self.inputs[0:batch_size_mid, 0:self.image_dims[0], int(self.image_dims[1] / 2):self.image_dims[1],
                                               0:self.image_dims[2]]
                _, _, class_encoded = self.discriminator2(
                    tf.transpose(self.input_right, [0, 3, 1, 2]),
                    num_classes=self.config.num_classes,
                    reuse=False, data_format='NCHW')
                self.class_onehot = tf.one_hot(tf.cast(tf.argmax(class_encoded, 1), dtype=tf.int32),
                                               self.config.num_classes,
                                               on_value=1., off_value=0., dtype=tf.float32)
                self.z = tf.concat([z_encoded, self.class_onehot], 1)

        if self.config.G_num == 2:
            self.G1 = self.generator1(self.z)
            self.G2 = self.generator2(self.z)
        else:
            self.G = self.generator(self.z)
        # self.D, self.D_logits = self.discriminator(self.inputs)
        # if self.config.G_num == 2:
        #     self.D_, self.D_logits_ = self.discriminator(tf.concat([self.G1, self.G2], 2), reuse=True)
        # else:
        #     self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # define loss
        # self.d_loss_real = tf.reduce_mean(
        #     networks.sigmoid_cross_entropy_with_logits(
        #         self.D_logits, tf.ones_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(
        #     networks.sigmoid_cross_entropy_with_logits(
        #         self.D_logits_, tf.zeros_like(self.D_)))
        # self.d_loss = self.d_loss_real + self.d_loss_fake

        # self.g_loss = tf.reduce_mean(
        #     networks.sigmoid_cross_entropy_with_logits(
        #         self.D_logits_, tf.ones_like(self.D_)))*self.config.stage2_g_loss
        #
        # # contexture loss
        # prob_a = tf.multiply(self.masks, self.inputs)
        # if self.config.G_num == 2:
        #     prob_b = tf.multiply(self.masks, tf.concat([self.G1, self.G2], 2))
        # else:
        #     prob_b = tf.multiply(self.masks, self.G)
        # self.c_loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=prob_a, logits=prob_b))*self.config.stage2_c_loss
        #
        # # l1 loss
        # self.l1_loss = tf.reduce_mean(tf.abs(prob_a - prob_b))*self.config.stage2_l1_loss

        # self.e_loss = self.g_loss+self.c_loss+self.l1_loss

        # optimizer

        # optimizer
        # annotation for testing, cancel annotation when training stage 2
        # if self.config.optim is "adam":
        #     self.e_optim = tf.train.AdamOptimizer(self.config.learning_rate,
        #                                           beta1=self.config.beta1).minimize(
        #         self.e_loss, var_list=self.encoder.var_list)
        # elif self.config.optim == "rmsprop":
        #     self.e_optim = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
        #         self.e_loss, var_list=self.encoder.var_list)

        # define sumary
        # self.z_sum = networks.histogram_summary("z", self.z)
        # self.inputs_sum = networks.image_summary("inputs", self.inputs)
        # self.input_left_sum = networks.image_summary("input_left", self.input_left)
        # self.masks_sum = networks.image_summary("masks", self.masks)
        #
        # self.d_sum = networks.histogram_summary("d", self.D)
        # self.d__sum = networks.histogram_summary("d_", self.D_)
        # if self.config.G_num == 2:
        #     self.G_sum = networks.image_summary("G", self.G2)
        # else:
        #     self.G_sum = networks.image_summary("G", self.G)

        # self.d_loss_real_sum = networks.scalar_summary("d_loss_real", self.d_loss_real)
        # self.d_loss_fake_sum = networks.scalar_summary("d_loss_fake", self.d_loss_fake)

        # self.c_loss_sum = networks.scalar_summary("c_loss", self.c_loss)
        #
        # self.g_loss_sum = networks.scalar_summary("g_loss", self.g_loss)
        # # self.d_loss_sum = networks.scalar_summary("d_loss", self.d_loss)
        # self.l1_loss_sum = networks.scalar_summary("l1_loss", self.l1_loss)
        #
        # self.e_loss_sum = networks.scalar_summary("e_loss", self.e_loss)

        # self.g_sum = networks.merge_summary([self.z_sum, self.d__sum, self.G_sum,
        #                         self.d_loss_fake_sum, self.g_loss_sum])
        # self.d_sum = networks.merge_summary([self.z_sum, self.d_sum,
        #                         self.d_loss_real_sum, self.d_loss_sum])
        # self.e_sum = networks.merge_summary([self.z_sum, self.inputs_sum, self.input_left_sum, self.masks_sum, self.G_sum,
        #                         self.g_loss_sum, self.c_loss_sum, self.l1_loss_sum, self.e_loss_sum])

        # if self.config.G_num == 2:
        #     self.saver = tf.train.Saver({v.op.name: v for v in self.generator1.var_list+self.generator2.var_list+self.discriminator.var_list})
        # else:
        #     self.saver = tf.train.Saver({v.op.name: v for v in self.generator.var_list+self.discriminator.var_list})

        self.saver2 = tf.train.Saver()

        utils.show_all_variables()

    def train2(self):
        self.build_model2()

        # load data
        data_path = os.path.join(self.config.data_dir,
                                 self.config.dataset,
                                 "stage2", "train",
                                 self.config.input_fname_pattern)
        self.data = glob(data_path)
        if len(self.data) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")
        if len(self.data) < self.config.batch_size:
            raise Exception(
                "[!] Entire dataset size is less than the configured batch_size")

        self.grayscale = (self.c_dim == 1)

        # init var
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # init summary writer
        self.writer = networks.SummaryWriter("./logs", self.sess.graph)

        # load model if exist
        counter = 1
        start_time = time.time()
        if self.config.E_stage1:
            could_load, checkpoint_counter = self.load(self.saver2, self.config.checkpoint_dir + "/stage1_AddE",
                                                       self.model_dir)
        else:
            could_load, checkpoint_counter = self.load(
                self.saver, self.config.checkpoint_dir+"/stage1", self.model_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # init mask by batch
        left_mask = tf.ones([self.config.batch_size, self.image_dims[0],
                             self.image_dims[1] / 2, self.image_dims[2]], dtype=tf.float32, name=None)
        right_mask = tf.zeros([self.config.batch_size, self.image_dims[0],
                               self.image_dims[1] / 2, self.image_dims[2]], dtype=tf.float32, name=None)
        batch_masks = tf.concat([left_mask, right_mask], 2)
        batch_masks = batch_masks.eval()

        # train
        for epoch in xrange(self.config.epoch):
            np.random.shuffle(self.data)
            batch_idxs = min(
                len(self.data), self.config.train_size) // self.config.batch_size

            for idx in xrange(0, int(batch_idxs)):
                # read image by batch
                batch_files = self.data[idx *
                                        self.config.batch_size: (idx+1)*self.config.batch_size]
                batch = [
                    utils.get_image(batch_file,
                                    input_height=self.config.input_height,
                                    input_width=self.config.input_width,
                                    resize_height=self.config.output_height,
                                    resize_width=self.config.output_width,
                                    crop=self.config.crop,
                                    grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(
                        np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                # z_tmp = self.sess.run(self.z, feed_dict={self.inputs: batch_images})
                # for i in range(self.config.batch_size):
                #     print(z_tmp[i])

                # Update E network
                _, summary_str = self.sess.run([self.e_optim, self.e_sum],
                                               feed_dict={self.inputs: batch_images, self.masks: batch_masks})
                self.writer.add_summary(summary_str, counter)

                errG = self.g_loss.eval({self.inputs: batch_images})
                errC = self.c_loss.eval(
                    {self.inputs: batch_images, self.masks: batch_masks})
                errL1 = self.l1_loss.eval(
                    {self.inputs: batch_images, self.masks: batch_masks})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, c_loss: %.8f, g_loss: %.8f, l1_loss: %.8f, e_loss: %.8f"
                      % (epoch, self.config.epoch, idx, batch_idxs,
                         time.time() - start_time, errC, errG, errL1, errC+errG+errL1))

                if np.mod(counter, self.config.save_checkpoint_frequency) == 2:
                    self.save(self.saver2, self.config.checkpoint_dir +
                              "/stage2", self.model_dir2, counter)

    def test2(self):
        self.build_model2()

        # load data
        if self.config.E_stage1:
            data_tmp = []
            if self.config.single_model == False:
                data_path = os.path.join(self.config.data_dir,
                                         self.config.dataset,
                                         "stage1", "sketch_instance", str(
                                             self.config.test_label),
                                         "*.png")
                data_tmp.extend(glob(data_path))
                data_path = os.path.join(self.config.data_dir,
                                         self.config.dataset,
                                         "stage1", "sketch_instance", str(
                                             self.config.test_label),
                                         "*.jpg")
                data_tmp.extend(glob(data_path))
            else:
                data_path = os.path.join(self.config.data_dir,
                                         self.config.dataset,
                                         "stage1", "test",
                                         "*.png")
                data_tmp.extend(glob(data_path))
                data_path = os.path.join(self.config.data_dir,
                                         self.config.dataset,
                                         "stage1", "test",
                                         "*.jpg")
                data_tmp.extend(glob(data_path))
        else:
            data_path = os.path.join(self.config.data_dir,
                                     self.config.dataset,
                                     "stage2", "test",
                                     self.config.input_fname_pattern)
            data_tmp = glob(data_path)
        self.data = sorted(data_tmp)

        if len(self.data) == 0:
            raise Exception("[!] No data found in '" + data_path + "'")
        if len(self.data) < self.config.batch_size and self.config.output_form is "batch":
            raise Exception(
                "[!] Entire dataset size is less than the configured batch_size")

        self.grayscale = (self.c_dim == 1)

        # init var
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # load model if exist
        counter = 1
        start_time = time.time()
        # test step 1 model which has Encoder

        if self.config.E_stage1:
            could_load, checkpoint_counter = self.load(
                self.saver2, self.config.checkpoint_dir+"/stage1_AddE", self.model_dir)
        else:
            could_load, checkpoint_counter = self.load(self.saver2, self.config.checkpoint_dir + "/stage2",
                                                       self.model_dir2)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        # test
        # np.random.shuffle(self.data)

        if self.config.output_form == "batch":
            # name saved in txt
            filename = self.config.sample_dir + '/stage1_AddE_specified/' + self.config.dataset + '/' + str(
                self.config.test_label) + '/' + self.model_dir + '.txt'
            file = open(filename, 'w')
            file.truncate()
            for i in range(len(self.data)):
                s = str(self.data[i])
                cropped = s.split('/')[-1]
                file.write(cropped + '\n')
            file.close()
            batch_size_tmp = self.config.batch_size
        else:
            batch_size_tmp = 1

        batch_idxs = min(
            len(self.data), self.config.train_size) // batch_size_tmp

        for idx in xrange(0, int(batch_idxs)):
            # read image by batch
            batch_files = self.data[idx*batch_size_tmp: (idx+1)*batch_size_tmp]
            batch = [
                utils.get_image(batch_file,
                                input_height=self.config.input_height,
                                input_width=self.config.input_width,
                                resize_height=self.config.output_height,
                                resize_width=self.config.output_width,
                                crop=self.config.crop,
                                grayscale=self.grayscale) for batch_file in batch_files]
            if self.grayscale:
                batch_images = np.array(batch).astype(
                    np.float32)[:, :, :, None]
            else:
                batch_images = np.array(batch).astype(np.float32)

            # generate images
            inputL = batch_images[:, :, 0:int(self.config.output_width / 2), :]
            if self.config.G_num == 2:
                outputL = self.sess.run(self.G1,
                                        feed_dict={self.inputs: batch_images})
                outputR = self.sess.run(self.G2,
                                        feed_dict={self.inputs: batch_images})
            else:
                results = self.sess.run(self.G,
                                        feed_dict={self.inputs: batch_images})
                outputL = results[:, :, 0:int(self.config.output_width / 2), :]
                outputR = results[:, :, int(
                    self.config.output_width / 2):self.config.output_width, :]
            # tmp = self.z.eval(feed_dict={self.inputs: batch_images})
            # a = tmp[1, :]
            if self.config.second_test == True:
                batch_images_second = np.append(outputL, outputL, axis=2)
                results_tmp = self.sess.run(self.G,
                                            feed_dict={self.inputs: batch_images_second})
                results = np.append(inputL, results, axis=2)
                results = np.append(results, results_tmp, axis=2)
            else:
                # result = np.hstack((batch_images[0], results[0]))
                # results = list(result)
                if self.config.output_combination == "inputL_outputR":
                    results = np.append(inputL, outputR, axis=2)
                elif self.config.output_combination == "outputL_inputR":
                    results = np.append(outputL, inputR, axis=2)
                elif self.config.output_combination == "outputR":
                    results = outputR
                else:
                    results = np.append(batch_images, outputL, axis=2)
                    results = np.append(results, outputR, axis=2)

            image_frame_dim = int(math.ceil(batch_size_tmp**.5))
            if self.config.E_stage1:
                if self.config.output_form == "batch":
                    utils.save_images(results, [image_frame_dim, image_frame_dim],
                                      self.config.sample_dir + '/stage1_AddE_specified/' + self.config.dataset + '/' + str(self.config.test_label) + '/' + self.model_dir + '__test_%s.png' % idx)
                else:
                    s2 = str(batch_files[0])
                    name = s2.split('/')[-1]
                    utils.save_images(results, [image_frame_dim, image_frame_dim],
                                      self.config.sample_dir + '/stage1_AddE_specified/' + self.config.dataset + '_singleTest/' + str(
                                          self.config.test_label) + '/' + name)

            else:
                utils.save_images(results, [image_frame_dim, image_frame_dim],
                                  self.config.sample_dir+'/stage2/'+self.model_dir2+'__test_%s.png' % idx)

            print("Test: [%4d/%4d]" % (idx, batch_idxs))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.config.dataset, self.config.batch_size,
            self.config.output_height, self.config.output_width)

    @property
    def model_dir2(self):
        return "{}_{}_{}_{}__{}_{}_{}".format(
            self.config.dataset, self.config.batch_size,
            self.config.output_height, self.config.output_width,
            self.config.stage2_g_loss, self.config.stage2_c_loss, self.config.stage2_l1_loss)

    def save(self, saver, checkpoint_dir, model_dir, step):
        print(" [*] Saving checkpoints...")
        model_name = 'DCGAN.model'
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        utils.makedirs(checkpoint_dir)

        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=step)

    def load(self, saver, checkpoint_dir, model_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        # print(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
