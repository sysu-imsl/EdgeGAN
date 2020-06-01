# -*- coding:utf8 -*-
# the entry of the project

import json
import os

import numpy as np
import tensorflow as tf
from numpy.random import seed

from edgegan.models import DCGAN
from edgegan.utils import makedirs, pp
from edgegan.utils.data import Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
                     "The name of dataset [class14_png_aug,]")
_FLAGS.DEFINE_string("input_fname_pattern", "*png",
                     "Glob pattern of filename of input images [*]")
_FLAGS.DEFINE_string("checkpoint_dir", None,
                     "Directory name to save the checkpoints [checkpoint]")
_FLAGS.DEFINE_string("logdir", None,
                     "Directory name to save the logs")
_FLAGS.DEFINE_string("dataroot", "./data", "Root directory of dataset [data]")
_FLAGS.DEFINE_integer("save_checkpoint_frequency", 500,
                      "frequency for saving checkpoint")
_FLAGS.DEFINE_boolean(
    "crop", False, "True for training, False for testing [False]")


# weight of loss
_FLAGS.DEFINE_float("stage1_zl_loss", 10.0, "weight of z l1 loss")

# multi class
_FLAGS.DEFINE_boolean("multiclasses", True, "if use focal loss")
_FLAGS.DEFINE_integer("num_classes", 14, "num of classes")
_FLAGS.DEFINE_string("SPECTRAL_NORM_UPDATE_OPS",
                     "spectral_norm_update_ops", "")

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
_FLAGS.DEFINE_integer("sizeOfIn_patch2", 128, "The size of input for D_patch2")
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
_FLAGS.DEFINE_integer("z_dim", 100, "dimension of random vector z")
FLAGS = _FLAGS.FLAGS


def make_outputs_dir(flags):
    makedirs(flags.outputsroot)
    makedirs(flags.checkpoint_dir)
    makedirs(flags.logdir)


def update_and_save_flags(flags):
    if flags.input_width is None:
        flags.input_width = flags.input_height
    if flags.output_width is None:
        flags.output_width = flags.output_height

    if not flags.multiclasses:
        flags.num_classes = None

    path = os.path.join(flags.outputsroot, flags.name)
    setattr(flags, 'checkpoint_dir', os.path.join(path, 'checkpoints'))
    setattr(flags, 'logdir', os.path.join(path, 'logs'))
    flag_dict = flags.flag_values_dict()
    with open(os.path.join(path, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=4)

    return flags


def main(_):
    phase = 'train'
    flags = update_and_save_flags(FLAGS)
    pp.pprint(flags.__flags)
    make_outputs_dir(flags)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    dataset_config = {
        'input_height': flags.input_height,
        'input_width': flags.input_width,
        'output_height': flags.output_height,
        'output_width': flags.output_width,
        'crop': flags.crop,
        'grayscale': False,
        'z_dim': flags.z_dim,
    }

    with tf.Session(config=run_config) as sess:
        dataset = Dataset(
            flags.dataroot, flags.dataset,
            flags.train_size, flags.batch_size,
            dataset_config, flags.num_classes, phase)
        dcgan = DCGAN(sess, flags, dataset, z_dim=flags.z_dim)
        dcgan.train()


if __name__ == '__main__':
    tf.app.run()
