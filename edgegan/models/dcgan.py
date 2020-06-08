from __future__ import division, print_function

import math
import os
import pickle
import sys
import time
from glob import glob

import numpy as np
import tensorflow as tf

import edgegan.nn.functional as F
from edgegan import nn, utils

from .classifier import Classifier
from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator


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
            self.joint_dis_dloss, var_list=self.joint_discriminator.var_list))
        self.register_optim_if('d_optim_patch2', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.image_dis_dloss, var_list=self.image_discriminator.var_list), self.config.use_image_discriminator)
        self.register_optim_if('d_optim_patch3', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.edge_dis_dloss, var_list=self.edge_discriminator.var_list), self.config.use_edge_discriminator)
        self.register_optim_if('d_optim2', tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
            self.loss_d_ac, var_list=self.classifier.var_list), self.config.multiclasses)
        self.register_optim_if(
            'g_optim', [
                tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
                    self.edge_gloss, var_list=self.edge_generator.var_list),
                tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(
                    self.image_gloss, var_list=self.image_generator.var_list)],
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
        self.edge_generator = Generator('G1', is_train=True,
                                        norm=self.config.G_norm,
                                        batch_size=self.config.batch_size,
                                        output_height=self.config.output_height,
                                        output_width=int(
                                            self.config.output_width/2),
                                        input_dim=self.gf_dim,
                                        output_dim=self.c_dim,
                                        use_resnet=self.config.if_resnet_g)
        self.image_generator = Generator('G2', is_train=True,
                                         norm=self.config.G_norm,
                                         batch_size=self.config.batch_size,
                                         output_height=self.config.output_height,
                                         output_width=int(
                                             self.config.output_width/2),
                                         input_dim=self.gf_dim,
                                         output_dim=self.c_dim,
                                         use_resnet=self.config.if_resnet_g)

        if self.config.multiclasses:
            self.classifier = Classifier(
                'D2', self.config.SPECTRAL_NORM_UPDATE_OPS)

        self.joint_discriminator = Discriminator('D', is_train=True,
                                                 norm=self.config.D_norm,
                                                 num_filters=self.df_dim,
                                                 use_resnet=self.config.if_resnet_d)

        if self.config.use_image_discriminator is True:
            self.image_discriminator = Discriminator('D_patch2', is_train=True,
                                                     norm=self.config.D_norm,
                                                     num_filters=self.df_dim,
                                                     use_resnet=self.config.if_resnet_d)

        if self.config.use_edge_discriminator is True:
            self.edge_discriminator = Discriminator('D_patch3', is_train=True,
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

        if self.config.multiclasses:
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
            _ = tf.image.resize_images(pictures, [self.config.edge_dis_size, self.config.edge_dis_size],
                                       method=2)

        if self.config.multiclasses:
            self.edge_output = self.edge_generator(self.z_onehot)
            self.image_output = self.image_generator(self.z_onehot)
        else:
            self.edge_output = self.edge_generator(self.z)
            self.image_output = self.image_generator(self.z)

        if self.config.multiclasses:
            def classify(inputs, reuse):
                _, _, result = self.classifier(
                    channel_first(inputs),
                    num_classes=self.config.num_classes,
                    labels=self.z[:, -1],
                    reuse=reuse, data_format='NCHW'
                )
                return result

            self.trueimage_class_output = classify(
                self.inputs[:, :, int(self.image_dims[1]/2):, :], reuse=False)
            self.fakeimage_class_output = classify(
                self.image_output, reuse=True)

        self.D, self.truejoint_dis_output = self.joint_discriminator(
            self.inputs)
        self.joint_output = tf.concat([self.edge_output, self.image_output], 2)
        self.D_, self.fakejoint_dis_output = self.joint_discriminator(
            self.joint_output, reuse=True)
        edges, pictures = split_inputs(self.inputs)

        if self.config.use_image_discriminator:
            self.resized_inputs = resize(pictures, self.config.image_dis_size)
            self.imageD, self.trueimage_dis_output = self.image_discriminator(
                self.resized_inputs)

            self.resized_image_output = resize(
                self.image_output, self.config.image_dis_size)
            self.imageDfake, self.fakeimage_dis_output = self.image_discriminator(
                self.resized_image_output, reuse=True)

        if self.config.use_edge_discriminator:
            self.resized_edges = resize(edges, self.config.edge_dis_size)

            delete_it_later()
            self.edgeD, self.trueedge_dis_output = self.edge_discriminator(
                self.resized_edges)

            self.resized_edge_output = resize(
                self.edge_output, self.config.edge_dis_size)
            self.edgeDfake, self.fakeedge_dis_output = self.edge_discriminator(
                self.resized_edge_output, reuse=True)

        self.z_recon, _, _ = self.encoder(self.edge_output)

    def define_losses(self):
        self.joint_dis_dloss = (
            F.discriminator_ganloss(self.fakejoint_dis_output, self.truejoint_dis_output) +
            penalty(
                self.joint_output, self.inputs, self.joint_discriminator,
                self.config.batch_size, self.config.lambda_gp
            )
        )
        self.joint_dis_gloss = F.generator_ganloss(self.fakejoint_dis_output)

        if self.config.use_image_discriminator:
            self.image_dis_dloss = (
                F.discriminator_ganloss(self.fakeimage_dis_output, self.trueimage_dis_output) +
                penalty(
                    self.resized_image_output, self.resized_inputs, self.image_discriminator,
                    self.config.batch_size, self.config.lambda_gp
                )
            )
            self.image_dis_gloss = F.generator_ganloss(
                self.fakeimage_dis_output)
        else:
            self.image_dis_dloss = 0
            self.image_dis_gloss = 0

        if self.config.use_edge_discriminator:
            self.edge_dis_dloss = (
                F.discriminator_ganloss(self.fakeedge_dis_output, self.trueedge_dis_output) +
                penalty(
                    self.resized_edge_output, self.resized_edges, self.edge_discriminator,
                    self.config.batch_size, self.config.lambda_gp
                )
            )
            self.edge_dis_gloss = F.generator_ganloss(self.fakeedge_dis_output)
        else:
            self.edge_dis_dloss = 0
            self.edge_dis_gloss = 0

        self.edge_gloss = (
            self.config.joint_dweight * self.joint_dis_gloss +
            self.config.edge_dweight * self.edge_dis_gloss
        )
        self.image_gloss = (
            self.config.joint_dweight * self.joint_dis_gloss +
            self.config.image_dweight * self.image_dis_gloss
        )

        # focal loss
        if self.config.multiclasses:
            self.loss_g_ac, self.loss_d_ac = F.get_acgan_loss_focal(
                self.trueimage_class_output, tf.cast(
                    self.z[:, -1], dtype=tf.int32),
                self.fakeimage_class_output, tf.cast(
                    self.z[:, -1], dtype=tf.int32),
                num_classes=self.config.num_classes)

            self.image_gloss += self.loss_g_ac
        else:
            self.loss_g_ac = 0
            self.loss_d_ac = 0

        z_target = self.z[:,
                          :self.z_dim] if self.config.multiclasses else self.z
        self.zl_loss = F.l1loss(
            z_target, self.z_recon,
            weight=self.config.stage1_zl_loss
        )

    def define_summaries(self):
        self.z_sum = nn.histogram_summary("z", self.z)
        self.inputs_sum = nn.image_summary("inputs", self.inputs)

        self.G1_sum = nn.image_summary("G1", self.edge_output)
        self.G2_sum = nn.image_summary("G2", self.image_output)

        self.g1_loss_sum = nn.scalar_summary("edge_gloss", self.edge_gloss)
        self.g2_loss_sum = nn.scalar_summary("image_gloss", self.image_gloss)

        self.g_loss_sum = nn.scalar_summary(
            "joint_dis_gloss", self.joint_dis_gloss)

        self.d_loss_sum = nn.scalar_summary(
            "joint_dis_dloss", self.joint_dis_dloss)

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

        if self.config.use_image_discriminator:
            self.d_patch2_sum = nn.histogram_summary(
                "imageD", self.imageD)
            self.d__patch2_sum = nn.histogram_summary(
                "imageDfake", self.imageDfake)
            self.resized_inputs_sum = nn.image_summary(
                "resized_inputs_image", self.resized_inputs)
            self.resized_G_sum = nn.image_summary(
                "resized_G_image", self.resized_image_output)
            self.d_loss_patch2_sum = nn.scalar_summary(
                "image_dis_dloss", self.image_dis_dloss)
            self.g_loss_patch2_sum = nn.scalar_summary(
                "image_dis_gloss", self.image_dis_gloss)
            self.g_sum = nn.merge_summary(
                [self.g_sum, self.d__patch2_sum, self.resized_G_sum, self.g_loss_patch2_sum])
            self.d_sum = nn.merge_summary(
                [self.d_sum, self.d_patch2_sum, self.resized_inputs_sum, self.d_loss_patch2_sum])

        if self.config.use_edge_discriminator:
            self.d_patch3_sum = nn.histogram_summary(
                "edgeD", self.edgeD)
            self.d__patch3_sum = nn.histogram_summary(
                "edgeDfake", self.edgeDfake)
            self.resized_inputs_p3_sum = nn.image_summary(
                "resized_inputs_p3_image", self.resized_edges)
            self.resized_G_p3_sum = nn.image_summary(
                "resized_G_p3_image", self.resized_edge_output)
            self.d_loss_patch3_sum = nn.scalar_summary(
                "edge_dis_dloss", self.edge_dis_dloss)
            self.g_loss_patch3_sum = nn.scalar_summary(
                "edge_dis_gloss", self.edge_dis_gloss)
            self.g_sum = nn.merge_summary(
                [self.g_sum, self.d__patch3_sum, self.resized_G_p3_sum, self.g_loss_patch3_sum])
            self.d_sum = nn.merge_summary(
                [self.d_sum, self.d_patch3_sum, self.resized_inputs_p3_sum, self.d_loss_patch3_sum])

    def build_train_model(self):
        self.build_networks()
        self.define_inputs()
        self.forward()
        self.define_losses()
        self.construct_optimizers()
        self.define_summaries()

        self.saver = tf.train.Saver()

        utils.show_all_variables()

    def train(self):

        def add_summary(images, z, counter):
            discriminator_summary = self.sess.run(
                self.d_sum, feed_dict={self.inputs: images, self.z: z})
            self.writer.add_summary(discriminator_summary, counter)
            generator_summary = self.sess.run(
                self.g_sum, feed_dict={self.inputs: images, self.z: z})
            self.writer.add_summary(generator_summary, counter)

        self.build_train_model()

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # init summary writer
        self.writer = nn.SummaryWriter(self.config.logdir, self.sess.graph)

        counter = 1
        start_time = time.time()
        loaded, checkpoint_counter = self.load(
            self.saver, self.config.checkpoint_dir)
        if loaded:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # train
        for epoch in range(self.config.epoch):
            self.dataset.shuffle()
            for idx in range(len(self.dataset)):
                batch_images, batch_z = self.dataset[idx]

                self.update_model(batch_images, batch_z)
                add_summary(batch_images, batch_z, counter)

                def evaluate(obj):
                    return getattr(obj, 'eval')(
                        {self.inputs: batch_images, self.z: batch_z})

                discriminator_err = evaluate(self.joint_dis_dloss)
                if self.config.use_image_discriminator:
                    discriminator_err += evaluate(self.image_dis_dloss)
                if self.config.use_edge_discriminator:
                    discriminator_err += evaluate(self.edge_dis_dloss)

                generator_err = evaluate(
                    self.edge_gloss) + evaluate(self.image_gloss)

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, joint_dis_dloss: %.8f, joint_dis_gloss: %.8f"
                      % (epoch, self.config.epoch, idx, len(self.dataset),
                         time.time() - start_time, 2 * discriminator_err, generator_err))
                if np.mod(counter, self.config.save_checkpoint_frequency) == 2:
                    self.save(self.saver, self.config.checkpoint_dir,
                              counter)

    def define_test_input(self):
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
        if self.config.multiclasses:
            if self.config.Test_singleLabel:
                batch_classes = np.full(
                    (self.config.batch_size, 1), self.config.test_label, dtype=np.float32)
                self.class_onehot = tf.one_hot(tf.cast(batch_classes[:, -1], dtype=tf.int32), self.config.num_classes,
                                               on_value=1., off_value=0., dtype=tf.float32)
                self.z = tf.concat([z_encoded, self.class_onehot], 1)

        self.edge_output = self.edge_generator(self.z)
        self.image_output = self.image_generator(self.z)

    def build_test_model(self):
        assert (not self.config.Test_allLabel) or (
            self.config.Test_singleLabel and self.config.test_label == 0)
        self.encoder = Encoder('E', is_train=True,
                               norm=self.config.E_norm,
                               image_size=self.config.input_height,
                               latent_dim=self.z_dim,
                               use_resnet=self.config.if_resnet_e)

        self.edge_generator = Generator('G1', is_train=False,
                                        norm=self.config.G_norm,
                                        batch_size=self.config.batch_size,
                                        output_height=self.config.output_height,
                                        output_width=int(
                                            self.config.output_width/2),
                                        input_dim=self.gf_dim,
                                        output_dim=self.c_dim,
                                        use_resnet=self.config.if_resnet_g)
        self.image_generator = Generator('G2', is_train=False,
                                         norm=self.config.G_norm,
                                         batch_size=self.config.batch_size,
                                         output_height=self.config.output_height,
                                         output_width=int(
                                             self.config.output_width/2),
                                         input_dim=self.gf_dim,
                                         output_dim=self.c_dim,
                                         use_resnet=self.config.if_resnet_g)

        self.define_test_input()

        self.saver = tf.train.Saver()

        utils.show_all_variables()

    def test(self):
        def pathsplit(path):
            path = os.path.normpath(path)
            return path.split(os.sep)

        def name_with_class(filename):
            splited = pathsplit(filename)
            return os.path.join(*splited[splited.index('test') + 1:])

        self.build_test_model()

        # init var
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        counter = 1
        start_time = time.time()
        loaded, checkpoint_counter = self.load(
            self.saver, self.config.checkpoint_dir)
        if loaded:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        for idx in range(len(self.dataset)):
            batch_images, filenames = self.dataset[idx]

            # generate images
            inputL = batch_images[:, :, 0:int(self.config.output_width / 2), :]
            outputL = self.sess.run(self.edge_output,
                                    feed_dict={self.inputs: batch_images})
            outputR = self.sess.run(self.image_output,
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

            assert results.shape[0] == len(filenames)
            for fname, img in zip(filenames, results):
                # name = fname.split('/')[- 1]
                name = name_with_class(fname)
                img = img[np.newaxis, ...]
                utils.save_images(
                    img, [1, 1],
                    os.path.join(
                        self.config.test_output_dir,
                        self.config.dataset, name,
                    )
                )

            print("Test: [%4d/%4d]" % (idx, len(self.dataset)))

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
