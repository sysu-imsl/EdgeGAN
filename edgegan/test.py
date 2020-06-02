import os

import numpy as np
import tensorflow as tf
from numpy.random import seed

from edgegan.models import DCGAN
from edgegan.utils import makedirs, pp
from edgegan.utils.data import Dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
phase = 'test'

seed(2333)
tf.set_random_seed(6666)

_FLAGS = tf.app.flags
_FLAGS.DEFINE_string("gpu", "0", "Gpu ID")
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
_FLAGS.DEFINE_string("test_output_dir", "test_output",
                     "Directory name to save the image samples [samples]")
_FLAGS.DEFINE_boolean(
    "crop", False, "True for training, False for testing [False]")


# setting of testing
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

# # multi class
_FLAGS.DEFINE_boolean("multiclasses", True, "if use focal loss")
_FLAGS.DEFINE_integer("num_classes", 14, "num of classes")

_FLAGS.DEFINE_string("type", "gpwgan", "gan type: [dcgan | wgan | gpwgan]")
_FLAGS.DEFINE_string("optim", "rmsprop", "optimizer type: [adam | rmsprop]")
_FLAGS.DEFINE_string("model", "old", "which base model(G and D): [old | new]")


_FLAGS.DEFINE_boolean("if_resnet_e", True, "if use resnet for E")
_FLAGS.DEFINE_boolean("if_resnet_g", False, "if use resnet for G")
_FLAGS.DEFINE_boolean("if_resnet_d", False, "if use resnet for origin D")
_FLAGS.DEFINE_string("E_norm", "instance",
                     "normalization options:[instance, batch, norm]")
_FLAGS.DEFINE_string("G_norm", "instance",
                     "normalization options:[instance, batch, norm]")
_FLAGS.DEFINE_string("D_norm", "instance",
                     "normalization options:[instance, batch, norm]")

FLAGS = _FLAGS.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

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
    setattr(flags, 'test_output_dir', os.path.join(path, 'test_output'))

    return flags


def create_dataset(flags):
    dataset_config = {
        'input_height': flags.input_height,
        'input_width': flags.input_width,
        'output_height': flags.output_height,
        'output_width': flags.output_width,
        'crop': flags.crop,
        'grayscale': False,
        'single_model': flags.single_model,
        'test_label': flags.test_label,
    }
    return Dataset(
        flags.dataroot, flags.dataset,
        flags.train_size, flags.batch_size,
        dataset_config, None, phase)


def main(_):
    flags = update_flags(FLAGS)
    pp.pprint(flags.__flags)
    make_outputs_dir(flags)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(sess, flags, None)
        if flags.Test_allLabel:
            for label in range(0, flags.num_classes):
                flags.test_label = label
                dcgan.dataset = create_dataset(flags)
                makedirs(flags.test_output_dir + "/stage1_AddE_specified/" + flags.dataset + '/' + str(
                    flags.test_label) + '/')
                dcgan.test()
        else:
            dcgan.dataset = create_dataset(flags)
            makedirs(flags.test_output_dir + "/stage1_AddE_specified/" + flags.dataset + '/' + str(
                flags.test_label) + '/')
            dcgan.test()


if __name__ == '__main__':
    tf.app.run()
