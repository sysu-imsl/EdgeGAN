# -*- coding:utf8 -*-
# defination of dcgan

from __future__ import division, print_function

import math
import os
import pickle
import sys
import time
from glob import glob

import numpy as np
import tensorflow as tf
from six.moves import xrange

import edgegan.nn.functional as F
from edgegan import nn, utils

from .classifier import Classifier
from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator

reload(sys)
sys.setdefaultencoding('utf8')


def allclose(a, b):
    if isinstance(a, np.ndarray):
        print('output error: {}'.format(np.mean(np.abs(a-b))))
        return np.mean(np.abs(a-b)) < 5e-4
    else:
        return abs((a-b) / a) < 0.01


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
        if extension(filename) == '.npy':
            return np.load(filename)
        elif extension(filename) == '.pkl':
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise NotImplementedError

    def enforce_exists(path):
        if not os.path.exists(path):
            print('missing loading item: {}'.format(path))
            raise ValueError

    checksum_path = utils.checksum_path
    result = []
    print(names)
    for name in names:
        path = os.path.join(checksum_path, name)
        enforce_exists(path)
        result.append(load(path))
    return result


def channel_first(input):
    return tf.transpose(input, [0, 3, 1, 2])


def random_blend(a, b, batchsize):
    alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
    alpha = alpha_dist.sample((batchsize, 1, 1, 1))
    return b + alpha * (a - b)


def penalty(synthesized, real, nn_func, batchsize, weight):
    assert callable(nn_func)
    interpolated = random_blend(synthesized, real, batchsize)
    inte_logit = nn_func(interpolated, reuse=True)
    return weight * F.gradient_penalty(inte_logit, interpolated)


class DCGAN(object):
    def __init__(self, sess, config, dataset,
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
        self.optimizers = []
        self.dataset = dataset

    def register_optim_if(self, name, optims, cond=True, repeat=1):
        if not cond:
            return
        optims = optims if isinstance(optims, list) else [optims, ]
        self.optimizers.append({
            'name': name,
            'optims': optims,
            'repeat': repeat,
        })

    def construct_optimizers(self):
        self.register_optim_if('d_optim', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.d_loss, var_list=self.discriminator.var_list))
        self.register_optim_if('d_optim_patch2', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.d_loss_patch2, var_list=self.discriminator_patch2.var_list), self.config.use_D_patch2)
        self.register_optim_if('d_optim_patch3', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.d_loss_patch3, var_list=self.discriminator_patch3.var_list), self.config.use_D_patch3)
        self.register_optim_if('d_optim2', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.loss_d_ac, var_list=self.classifier.var_list), self.config.if_focal_loss)
        self.register_optim_if(
            'g_optim', [
                tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
                    self.g1_loss, var_list=self.generator1.var_list),
                tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
                    self.g2_loss, var_list=self.generator2.var_list)],
            repeat=2
        )
        self.register_optim_if('e_optim', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.zl_loss, var_list=self.encoder.var_list))

    def update_model(self, images, z):
        for optim_param in self.optimizers:
            optims, repeat = optim_param['optims'], optim_param['repeat']
            for _ in range(repeat):
                _ = self.sess.run(optims, {self.inputs: images, self.z: z})

    def build_networks(self):
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

        if self.config.if_focal_loss:
            self.classifier = Classifier(
                'D2', self.config.SPECTRAL_NORM_UPDATE_OPS)

        self.discriminator = Discriminator('D', is_train=True,
                                           norm=self.config.D_norm,
                                           num_filters=self.df_dim,
                                           use_resnet=self.config.if_resnet_d)

        if self.config.use_D_patch2 is True:
            self.discriminator_patch2 = Discriminator('D_patch2', is_train=True,
                                                      norm=self.config.D_norm,
                                                      num_filters=self.df_dim,
                                                      use_resnet=self.config.if_resnet_d)

        if self.config.use_D_patch3 is True:
            self.discriminator_patch3 = Discriminator('D_patch3', is_train=True,
                                                      norm=self.config.D_norm,
                                                      num_filters=self.df_dim,
                                                      use_resnet=self.config.if_resnet_d)

        self.encoder = Encoder('E', is_train=True,
                               norm=self.config.E_norm,
                               image_size=self.config.input_height,
                               latent_dim=self.z_dim,
                               use_resnet=self.config.if_resnet_e)

    def define_inputs(self):
        image_size = (
            [self.config.output_height, self.config.output_width, ] if self.config.crop
            else [self.config.input_height, self.config.input_width, ]
        )
        self.image_dims = image_size + [self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.config.batch_size] + self.image_dims, name='real_images')

        if self.config.if_focal_loss:
            self.z = tf.placeholder(
                tf.float32, [None, self.z_dim+1], name='z')
            class_onehot = tf.one_hot(
                tf.cast(self.z[:, -1], dtype=tf.int32),
                self.config.num_classes,
                on_value=1., off_value=0., dtype=tf.float32
            )
            self.z_onehot = tf.concat(
                [self.z[:, 0:self.z_dim], class_onehot], 1)
        else:
            self.z = tf.placeholder(
                tf.float32, [None, self.z_dim], name='z')

    def forward(self):
        def split_inputs(inputs):
            begin = int(self.config.output_width / 2)
            end = self.config.output_width
            return (
                inputs[:, :, :begin, :],
                inputs[:, :, begin: end:]
            )

        def resize(inputs, size):
            return tf.image.resize_images(
                inputs, [size, size], method=2)

        def delete_it_later():
            pictures = self.inputs[:, :, int(
                self.config.output_width / 2):self.config.output_width, :]
            _ = tf.image.resize_images(pictures, [self.config.sizeOfIn_patch3, self.config.sizeOfIn_patch3],
                                       method=2)

        if self.config.if_focal_loss:
            self.G1 = self.generator1(self.z_onehot)
            self.G2 = self.generator2(self.z_onehot)
        else:
            self.G1 = self.generator1(self.z)
            self.G2 = self.generator2(self.z)

        if self.config.if_focal_loss:
            def classify(inputs, reuse):
                _, _, result = self.classifier(
                    channel_first(inputs),
                    num_classes=self.config.num_classes,
                    labels=self.z[:, -1],
                    reuse=reuse, data_format='NCHW'
                )
                return result

            self.D_logits2 = classify(
                self.inputs[:, :, int(self.image_dims[1]/2):, :], reuse=False)
            self.D_logits2_ = classify(self.G2, reuse=True)

        self.D, self.D_logits = self.discriminator(self.inputs)
        self.G_all = tf.concat([self.G1, self.G2], 2)
        self.D_, self.D_logits_ = self.discriminator(self.G_all, reuse=True)
        edges, pictures = split_inputs(self.inputs)

        if self.config.use_D_patch2:
            self.resized_inputs = resize(pictures, self.config.sizeOfIn_patch2)
            self.patch2_D, self.patch2_D_logits = self.discriminator_patch2(
                self.resized_inputs)

            self.resized_G2_p2 = resize(self.G2, self.config.sizeOfIn_patch2)
            self.patch2_D_, self.patch2_D_logits_ = self.discriminator_patch2(
                self.resized_G2_p2, reuse=True)

        if self.config.use_D_patch3:
            self.resized_inputs_p3 = resize(edges, self.config.sizeOfIn_patch3)

            delete_it_later()
            self.patch3_D, self.patch3_D_logits = self.discriminator_patch3(
                self.resized_inputs_p3)

            self.resized_G1_p3 = resize(self.G1, self.config.sizeOfIn_patch3)
            self.patch3_D_, self.patch3_D_logits_ = self.discriminator_patch3(
                self.resized_G1_p3, reuse=True)

        self.z_recon, _, _ = self.encoder(self.G1)

    def define_losses(self):
        self.d_loss = (
            F.discriminator_ganloss(self.D_logits_, self.D_logits) +
            penalty(
                self.G_all, self.inputs, self.discriminator,
                self.config.batch_size, self.config.lambda_gp
            )
        )
        self.g_loss = F.generator_ganloss(self.D_logits_)

        if self.config.use_D_patch2:
            self.d_loss_patch2 = (
                F.discriminator_ganloss(self.patch2_D_logits_, self.patch2_D_logits) +
                penalty(
                    self.resized_G2_p2, self.resized_inputs, self.discriminator_patch2,
                    self.config.batch_size, self.config.lambda_gp
                )
            )
            self.g_loss_patch2 = F.generator_ganloss(self.patch2_D_logits_)
        else:
            self.d_loss_patch2 = 0
            self.g_loss_patch2 = 0

        if self.config.use_D_patch3:
            self.d_loss_patch3 = (
                F.discriminator_ganloss(self.patch3_D_logits_, self.patch3_D_logits) +
                penalty(
                    self.resized_G1_p3, self.resized_inputs_p3, self.discriminator_patch3,
                    self.config.batch_size, self.config.lambda_gp
                )
            )
            self.g_loss_patch3 = F.generator_ganloss(self.patch3_D_logits_)
        else:
            self.d_loss_patch3 = 0
            self.g_loss_patch3 = 0

        self.g1_loss = (
            self.config.D_origin_loss * self.g_loss +
            self.config.D_patch3_loss * self.g_loss_patch3
        )
        self.g2_loss = (
            self.config.D_origin_loss * self.g_loss +
            self.config.D_patch2_loss * self.g_loss_patch2
        )

        # focal loss
        if self.config.if_focal_loss:
            self.loss_g_ac, self.loss_d_ac = F.get_acgan_loss_focal(
                self.D_logits2, tf.cast(self.z[:, -1], dtype=tf.int32),
                self.D_logits2_, tf.cast(self.z[:, -1], dtype=tf.int32),
                num_classes=self.config.num_classes)

            self.g2_loss += self.loss_g_ac
        else:
            self.loss_g_ac = 0
            self.loss_d_ac = 0

        z_target = self.z[:,
                          :self.z_dim] if self.config.if_focal_loss else self.z
        self.zl_loss = F.l1loss(
            z_target, self.z_recon,
            weight=self.config.stage1_zl_loss
        )

    def define_summaries(self):
        self.z_sum = nn.histogram_summary("z", self.z)
        self.inputs_sum = nn.image_summary("inputs", self.inputs)

        self.G1_sum = nn.image_summary("G1", self.G1)
        self.G2_sum = nn.image_summary("G2", self.G2)

        self.g1_loss_sum = nn.scalar_summary("g1_loss", self.g1_loss)
        self.g2_loss_sum = nn.scalar_summary("g2_loss", self.g2_loss)

        self.g_loss_sum = nn.scalar_summary("g_loss", self.g_loss)

        self.d_loss_sum = nn.scalar_summary("d_loss", self.d_loss)

        self.zl_loss_sum = nn.scalar_summary("zl_loss", self.zl_loss)

        self.loss_g_ac_sum = nn.scalar_summary(
            "loss_g_ac", self.loss_g_ac)
        self.loss_d_ac_sum = nn.scalar_summary(
            "loss_d_ac", self.loss_d_ac)

        self.g_sum = nn.merge_summary([self.z_sum, self.G1_sum, self.G2_sum,
                                       self.zl_loss_sum, self.g_loss_sum,
                                       self.loss_g_ac_sum, self.g1_loss_sum, self.g2_loss_sum])
        self.d_sum = nn.merge_summary([self.z_sum, self.inputs_sum,
                                       self.d_loss_sum, self.loss_d_ac_sum])
        self.d_sum_tmp = nn.histogram_summary("d", self.D)
        self.d__sum_tmp = nn.histogram_summary("d_", self.D_)
        self.g_sum = nn.merge_summary([self.g_sum, self.d__sum_tmp])
        self.d_sum = nn.merge_summary([self.d_sum, self.d_sum_tmp])

        if self.config.use_D_patch2:
            self.d_patch2_sum = nn.histogram_summary(
                "patch2_d", self.patch2_D)
            self.d__patch2_sum = nn.histogram_summary(
                "patch2_d_", self.patch2_D_)
            self.resized_inputs_sum = nn.image_summary(
                "resized_inputs_image", self.resized_inputs)
            self.resized_G_sum = nn.image_summary(
                "resized_G_image", self.resized_G2_p2)
            self.d_loss_patch2_sum = nn.scalar_summary(
                "d_loss_patch2", self.d_loss_patch2)
            self.g_loss_patch2_sum = nn.scalar_summary(
                "g_loss_patch2", self.g_loss_patch2)
            self.g_sum = nn.merge_summary(
                [self.g_sum, self.d__patch2_sum, self.resized_G_sum, self.g_loss_patch2_sum])
            self.d_sum = nn.merge_summary(
                [self.d_sum, self.d_patch2_sum, self.resized_inputs_sum, self.d_loss_patch2_sum])

        if self.config.use_D_patch3:
            self.d_patch3_sum = nn.histogram_summary(
                "patch3_d", self.patch3_D)
            self.d__patch3_sum = nn.histogram_summary(
                "patch3_d_", self.patch3_D_)
            self.resized_inputs_p3_sum = nn.image_summary(
                "resized_inputs_p3_image", self.resized_inputs_p3)
            self.resized_G_p3_sum = nn.image_summary(
                "resized_G_p3_image", self.resized_G1_p3)
            self.d_loss_patch3_sum = nn.scalar_summary(
                "d_loss_patch3", self.d_loss_patch3)
            self.g_loss_patch3_sum = nn.scalar_summary(
                "g_loss_patch3", self.g_loss_patch3)
            self.g_sum = nn.merge_summary(
                [self.g_sum, self.d__patch3_sum, self.resized_G_p3_sum, self.g_loss_patch3_sum])
            self.d_sum = nn.merge_summary(
                [self.d_sum, self.d_patch3_sum, self.resized_inputs_p3_sum, self.d_loss_patch3_sum])

    def build_model1(self):
        self.build_networks()
        self.define_inputs()
        self.forward()
        self.define_losses()
        self.construct_optimizers()
        self.define_summaries()

        self.saver = tf.train.Saver(
            {v.op.name: v for v in self.generator1.var_list + self.generator2.var_list})
        self.saver2 = tf.train.Saver()

        utils.show_all_variables()

    def train(self):

        def add_summary(images, z, counter):
            discriminator_summary = self.sess.run(
                self.d_sum, feed_dict={self.inputs: images, self.z: z})
            self.writer.add_summary(discriminator_summary, counter)
            generator_summary = self.sess.run(
                self.g_sum, feed_dict={self.inputs: images, self.z: z})
            self.writer.add_summary(generator_summary, counter)

        self.build_model1()

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # init summary writer
        self.writer = nn.SummaryWriter(self.config.logdir, self.sess.graph)

        counter = 1
        start_time = time.time()
        loaded, checkpoint_counter = self.load(
            self.saver2, self.config.checkpoint_dir, self.model_dir)
        if loaded:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            raise ValueError
            print(" [!] Load failed...")

        # train
        for epoch in xrange(1):
            self.dataset.shuffle()
            for idx in xrange(1):
                (batch_files, batch_images, batch_z) = checksum_load(
                    'batch_files.pkl', 'batch_images.npy', 'batch_z.npy')

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

                img_from_dataset, z_from_dataset = self.dataset[0]
                assert allclose(batch_images, img_from_dataset)
                assert batch_z.shape == z_from_dataset.shape

                self.update_model(batch_images, batch_z)
                add_summary(batch_images, batch_z, counter)

                def evaluate(obj):
                    return getattr(obj, 'eval')(
                        {self.inputs: batch_images, self.z: batch_z})

                discriminator_err = evaluate(self.d_loss)
                if self.config.use_D_patch2:
                    discriminator_err += evaluate(self.d_loss_patch2)
                if self.config.use_D_patch3:
                    discriminator_err += evaluate(self.d_loss_patch3)

                generator_err = evaluate(self.g1_loss) + evaluate(self.g2_loss)

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, self.config.epoch, idx, len(self.dataset),
                         time.time() - start_time, 2 * discriminator_err, generator_err))

                outputL = self.sess.run(self.G1,
                                        feed_dict={self.z: batch_z, self.inputs: batch_images})

                restore_outputL, restore_errD_fake, restore_errD_real, restore_errG = checksum_load(
                    "outputL.npy", "errD_fake.pkl", "errD_real.pkl", "errG.pkl",)
                assert allclose(restore_outputL, outputL)
                assert allclose(restore_errD_fake, discriminator_err)
                assert allclose(restore_errG, generator_err)
                print('assert successed!')
                # self.save(self.saver2, self.config.checkpoint_dir, self.model_dir, counter)
                exit()

    def define_test_input(self):
        # define inputs
        if self.config.crop:
            self.image_dims = [self.config.output_height, self.config.output_width,
                               self.c_dim]
        else:
            self.image_dims = [self.config.input_height, self.config.input_width,
                               self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.config.batch_size] + self.image_dims, name='real_images')

        self.masks = tf.placeholder(
            tf.float32, [self.config.batch_size] + self.image_dims, name='mask_images')

        self.input_left = self.inputs[0:self.config.batch_size, 0:self.image_dims[0], 0:int(
            self.image_dims[1]/2), 0:self.image_dims[2]]
        z_encoded, z_encoded_mu, z_encoded_log_sigma = self.encoder(
            self.input_left)
        self.z = z_encoded
        if self.config.if_focal_loss:
            if self.config.Test_singleLabel:
                batch_classes = np.full(
                    (self.config.batch_size, 1), self.config.test_label, dtype=np.float32)
                # batch_z = np.concatenate((z_encoded, batch_classes), axis=1)
                self.class_onehot = tf.one_hot(tf.cast(batch_classes[:, -1], dtype=tf.int32), self.config.num_classes,
                                               on_value=1., off_value=0., dtype=tf.float32)
                self.z = tf.concat([z_encoded, self.class_onehot], 1)

        self.G1 = self.generator1(self.z)
        self.G2 = self.generator2(self.z)

    def build_model2(self):
        assert (not self.config.Test_allLabel) or (
            self.config.Test_singleLabel and self.config.test_label == 0)
        self.encoder = Encoder('E', is_train=True,
                               norm=self.config.E_norm,
                               image_size=self.config.input_height,
                               latent_dim=self.z_dim,
                               use_resnet=self.config.if_resnet_e)

        self.generator1 = Generator('G1', is_train=False,
                                    norm=self.config.G_norm,
                                    batch_size=self.config.batch_size,
                                    output_height=self.config.output_height,
                                    output_width=int(
                                        self.config.output_width/2),
                                    input_dim=self.gf_dim,
                                    output_dim=self.c_dim,
                                    use_resnet=self.config.if_resnet_g)
        self.generator2 = Generator('G2', is_train=False,
                                    norm=self.config.G_norm,
                                    batch_size=self.config.batch_size,
                                    output_height=self.config.output_height,
                                    output_width=int(
                                        self.config.output_width/2),
                                    input_dim=self.gf_dim,
                                    output_dim=self.c_dim,
                                    use_resnet=self.config.if_resnet_g)

        self.define_test_input()

        self.saver2 = tf.train.Saver()

        utils.show_all_variables()

    def test2(self):
        self.build_model2()

        # load data
        data_tmp = []
        if self.config.single_model == False:
            data_path = os.path.join(self.config.dataroot,
                                     self.config.dataset,
                                     "stage1", "sketch_instance", str(
                                         self.config.test_label),
                                     "*.png")
            data_tmp.extend(glob(data_path))
            data_path = os.path.join(self.config.dataroot,
                                     self.config.dataset,
                                     "stage1", "sketch_instance", str(
                                         self.config.test_label),
                                     "*.jpg")
            data_tmp.extend(glob(data_path))
        else:
            data_path = os.path.join(self.config.dataroot,
                                     self.config.dataset,
                                     "stage1", "test",
                                     "*.png")
            data_tmp.extend(glob(data_path))
            data_path = os.path.join(self.config.dataroot,
                                     self.config.dataset,
                                     "stage1", "test",
                                     "*.jpg")
            data_tmp.extend(glob(data_path))

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
        could_load, checkpoint_counter = self.load(
            self.saver2, self.config.checkpoint_dir, self.model_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

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

            recon_batch_images = self.dataset[idx]
            assert allclose(recon_batch_images, batch_images)
            print('dataset assert successed!')

            # generate images
            inputL = batch_images[:, :, 0:int(self.config.output_width / 2), :]
            outputL = self.sess.run(self.G1,
                                    feed_dict={self.inputs: batch_images})
            outputR = self.sess.run(self.G2,
                                    feed_dict={self.inputs: batch_images})

            if self.config.output_combination == "inputL_outputR":
                results = np.append(inputL, outputR, axis=2)
            elif self.config.output_combination == "outputL_inputR":
                results = np.append(outputL, inputR, axis=2)
            elif self.config.output_combination == "outputR":
                results = outputR
            else:
                results = np.append(batch_images, outputL, axis=2)
                results = np.append(results, outputR, axis=2)

            recon_results = checksum_load('test2_reuslt.npy')
            assert allclose(results, recon_results)
            print('assertion successed!')
            exit(0)

            image_frame_dim = int(math.ceil(batch_size_tmp**.5))
            if self.config.output_form == "batch":
                utils.save_images(results, [image_frame_dim, image_frame_dim],
                                  self.config.sample_dir + '/stage1_AddE_specified/' + self.config.dataset + '/' + str(self.config.test_label) + '/' + self.model_dir + '__test_%s.png' % idx)
            else:
                s2 = str(batch_files[0])
                name = s2.split('/')[-1]
                utils.save_images(results, [image_frame_dim, image_frame_dim],
                                  self.config.sample_dir + '/stage1_AddE_specified/' + self.config.dataset + '_singleTest/' + str(
                    self.config.test_label) + '/' + name)

            print("Test: [%4d/%4d]" % (idx, batch_idxs))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.config.dataset, self.config.batch_size,
            self.config.output_height, self.config.output_width)

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
