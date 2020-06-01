# -*- coding:utf8 -*-
# the entry of the project

import os

import numpy as np
import tensorflow as tf
from numpy.random import seed

from edgegan.models import DCGAN
from edgegan.utils import makedirs, pp

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


seed(2333)
tf.set_random_seed(6666)

_FLAGS = tf.app.flags
_FLAGS.DEFINE_string("name", "edgegan", "Folder for all outputs")
_FLAGS.DEFINE_string("outputsroot", "outputs", "Outputs root")
_FLAGS.DEFINE_integer("epoch", 100, "Epoch to train [25]")
_FLAGS.DEFINE_float("learning_rate", 0.0002,
                    "Learning rate of for adam [0.0002]")
_FLAGS.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
_FLAGS.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
_FLAGS.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
_FLAGS.DEFINE_integer(
    "input_height", 64, "The size of image to use (will be center cropped). [108]")
_FLAGS.DEFINE_integer(
    "input_width", 128, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
_FLAGS.DEFINE_integer("output_height", 64,
                      "The size of the output images to produce [64]")
_FLAGS.DEFINE_integer("output_width", 128,
                      "The size of the output images to produce. If None, same value as output_height [None]")
_FLAGS.DEFINE_string("dataset", "class14_png_aug",
                     "The name of dataset [celebA, mnist, lsun]")
_FLAGS.DEFINE_string("input_fname_pattern", "*png",
                     "Glob pattern of filename of input images [*]")
_FLAGS.DEFINE_string("checkpoint_dir", None,
                     "Directory name to save the checkpoints [checkpoint]")
_FLAGS.DEFINE_string("logdir", None,
                     "Directory name to save the logs")
_FLAGS.DEFINE_string("dataroot", "./data", "Root directory of dataset [data]")
_FLAGS.DEFINE_string("test_output_dir", "samples_gpwgan_instanceEGD_noOriginD_patch2_128_patch3_128_patchGAN_insN_wgan_2G",
                     "Directory name to save the image samples [samples]")
_FLAGS.DEFINE_boolean(
    "train", False, "True for training, False for testing [False]")
_FLAGS.DEFINE_integer("save_checkpoint_frequency", 500,
                      "frequency for saving checkpoint")
_FLAGS.DEFINE_boolean(
    "crop", False, "True for training, False for testing [False]")


# setting of testing
_FLAGS.DEFINE_boolean("Random_test", False,
                      "IS effect when E_stage1 is True.True for testing random z, else for input images")
_FLAGS.DEFINE_boolean("Test_singleLabel", True,
                      "IS effect when Random_test is True or False.True for testing single label. For multi-class model")
_FLAGS.DEFINE_integer(
    "test_label", 3, "symbol of class, is effect when E_stage1 and Test_singleLabel are true, Random_test is false")
_FLAGS.DEFINE_boolean("Test_allLabel", True,
                      "Highest priority, True for testing all label, Test_singleLabel should be True. For multi-class model")
_FLAGS.DEFINE_boolean("single_model", False,
                      "True for testing single-class model")
_FLAGS.DEFINE_string("output_form", "batch",
                     "The format of output image: batch or single")
_FLAGS.DEFINE_string("output_combination", "full",
                     "The combination of output image: full(input+output), inputL_outputR(the left of input combine the right of output),outputL_inputR, outputR")

# weight of loss
_FLAGS.DEFINE_float("stage2_g_loss", 0.0, "weight of g loss")
_FLAGS.DEFINE_float("stage2_c_loss", 0.0, "weight of contexture loss")
_FLAGS.DEFINE_float("stage2_l1_loss", 0.0, "weight of l1 loss")
_FLAGS.DEFINE_float("stage1_zl_loss", 10.0, "weight of z l1 loss")

# multi class
_FLAGS.DEFINE_boolean("if_focal_loss", True, "if use focal loss")
_FLAGS.DEFINE_integer("num_classes", 14, "num of classes")
_FLAGS.DEFINE_string("SPECTRAL_NORM_UPDATE_OPS",
                     "spectral_norm_update_ops", "")

_FLAGS.DEFINE_string("type", "gpwgan", "gan type: [dcgan | wgan | gpwgan]")
_FLAGS.DEFINE_string("optim", "rmsprop", "optimizer type: [adam | rmsprop]")
_FLAGS.DEFINE_string("model", "old", "which base model(G and D): [old | new]")


_FLAGS.DEFINE_boolean("if_resnet_e", True, "if use resnet for E")
_FLAGS.DEFINE_boolean("if_resnet_g", False, "if use resnet for G")
_FLAGS.DEFINE_boolean("if_resnet_d", False, "if use resnet for origin D")
_FLAGS.DEFINE_float("lambda_gp", 10.0,
                    "if 'gpwgan' is chosen the corresponding lambda must be filled")
_FLAGS.DEFINE_float("clamp_lower", -0.01,
                    "if 'wgan' is chosen the corresponding lambda must be filled, the upper bound of parameters in disc")
_FLAGS.DEFINE_float("clamp_upper", 0.01,
                    "if 'wgan' is chosen the corresponding lambda must be filled, the upper bound of parameters in disc")

_FLAGS.DEFINE_string("E_norm", "instance",
                     "normalization options:[instance, batch, norm]")
_FLAGS.DEFINE_string("G_norm", "instance",
                     "normalization options:[instance, batch, norm]")
_FLAGS.DEFINE_string("D_norm", "instance",
                     "normalization options:[instance, batch, norm]")


_FLAGS.DEFINE_boolean("use_D_patch2", True,
                      "True for using patch discriminator, modify the size of input of discriminator")
# flags.DEFINE_integer("scale_num", 2, "num of of multi-discriminator")
_FLAGS.DEFINE_integer("sizeOfIn_patch2", 128, "The size of input for D_patch2")

# flags.DEFINE_integer("scale_num", 2, "num of of multi-discriminator")
_FLAGS.DEFINE_integer("sizeOfIn_patch2_2", 256,
                      "The size of input for D_patch2_2")

_FLAGS.DEFINE_boolean("use_D_patch3", True,
                      "True for using patch discriminator, modify the size of input of discriminator, user for edge discriminator when G_num == 2")
_FLAGS.DEFINE_integer("sizeOfIn_patch3", 128, "The size of input for D_patch2")


_FLAGS.DEFINE_float("D_origin_loss", 1.0,
                    "weight of origin discriminative loss, is ineffective when use_D_origin is false")
_FLAGS.DEFINE_float("D_patch2_loss", 1.0,
                    "weight of patch discriminative loss, is ineffective when use_D_patch2 is false")
_FLAGS.DEFINE_float("D_patch3_loss", 1.0,
                    "weight of patch discriminative loss, is ineffective when use_D_patch3 is false")
FLAGS = _FLAGS.FLAGS


def make_outputs_dir(flags):
    makedirs(flags.outputsroot)
    makedirs(flags.checkpoint_dir)
    makedirs(flags.logdir)


def update_flags(flags):
    if flags.input_width is None:
        flags.input_width = flags.input_height
    if flags.output_width is None:
        flags.output_width = flags.output_height

    path = os.path.join(flags.outputsroot, flags.name)
    setattr(flags, 'checkpoint_dir', os.path.join(path, 'checkpoints'))
    setattr(flags, 'logdir', os.path.join(path, 'logs'))

    return flags


def main(_):
    flags = update_flags(FLAGS)
    pp.pprint(flags.__flags)
    make_outputs_dir(flags)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(sess, flags, None)

        if flags.train:
            dcgan.train()
        else:
            if flags.Test_allLabel:
                for label in xrange(0, flags.num_classes):
                    flags.test_label = label
                    if flags.output_form is "batch":
                        makedirs(flags.test_output_dir + "/stage1_AddE_specified/" + flags.dataset + '/' + str(
                            flags.test_label) + '/')
                    else:
                        makedirs(flags.test_output_dir + "/stage1_AddE_specified/" + flags.dataset + '_singleTest/' + str(
                            flags.test_label) + '/')
                    dcgan.test2()
            elif not flags.Random_test:
                if flags.output_form is "batch":
                    makedirs(flags.test_output_dir + "/stage1_AddE_specified/" + flags.dataset + '/' + str(
                        flags.test_label) + '/')
                else:
                    makedirs(flags.test_output_dir + "/stage1_AddE_specified/" + flags.dataset + '_singleTest/' + str(
                        flags.test_label) + '/')
                dcgan.test2()
            elif flags.Random_test and flags.Test_singleLabel:
                dcgan.test1(flags.num_classes * flags.batch_size)
            else:
                dcgan.test1(10*flags.batch_size)


if __name__ == '__main__':
    tf.app.run()
