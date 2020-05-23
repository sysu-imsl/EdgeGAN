# -*- coding:utf8 -*-
# the entry of the project

import os

import numpy as np
import scipy.misc
import tensorflow as tf
from numpy.random import seed

from edgegan.nn import DCGAN
from edgegan.utils import makedirs, pp, show_all_variables, to_json

os.environ['CUDA_VISIBLE_DEVICES']='0'


seed(2333)
tf.set_random_seed(6666)

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 64, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 128, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 128, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "class14_png_aug", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_gpwgan_instanceEGD_noOriginD_patch2_128_patch3_128_patchGAN_insN_wgan_2G", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples_gpwgan_instanceEGD_noOriginD_patch2_128_patch3_128_patchGAN_insN_wgan_2G", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("save_checkpoint_frequency", 500, "frequency for saving checkpoint")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")

flags.DEFINE_integer("G_num", 2, "setting 2 generators for edge and image generation seprately")

# setting of testing
flags.DEFINE_boolean("Random_test", False, "IS effect when E_stage1 is True.True for testing random z, else for input images")
flags.DEFINE_boolean("Test_singleLabel", True, "IS effect when Random_test is True or False.True for testing single label. For multi-class model")
flags.DEFINE_integer("test_label", 3, "symbol of class, is effect when E_stage1 and Test_singleLabel are true, Random_test is false")
flags.DEFINE_boolean("Test_allLabel", True, "Highest priority, True for testing all label, Test_singleLabel should be True. For multi-class model")
flags.DEFINE_boolean("single_model", False, "True for testing single-class model")
flags.DEFINE_string("output_form", "batch", "The format of output image: batch or single")
flags.DEFINE_string("output_combination", "full", "The combination of output image: full(input+output), inputL_outputR(the left of input combine the right of output),outputL_inputR, outputR")

# weight of loss
flags.DEFINE_float("stage2_g_loss", 0.0, "weight of g loss")
flags.DEFINE_float("stage2_c_loss", 0.0, "weight of contexture loss")
flags.DEFINE_float("stage2_l1_loss", 0.0, "weight of l1 loss")
flags.DEFINE_float("stage1_zl_loss", 10.0, "weight of z l1 loss")

# multi class
flags.DEFINE_boolean("if_focal_loss", True, "if use focal loss")
flags.DEFINE_integer("num_classes", 14, "num of classes")
flags.DEFINE_string("SPECTRAL_NORM_UPDATE_OPS", "spectral_norm_update_ops", "")

flags.DEFINE_string("type", "gpwgan", "gan type: [dcgan | wgan | gpwgan]")
flags.DEFINE_string("optim", "rmsprop", "optimizer type: [adam | rmsprop]")
flags.DEFINE_string("model", "old", "which base model(G and D): [old | new]")
flags.DEFINE_boolean("if_resnet_e", True, "if use resnet for E")
flags.DEFINE_boolean("if_resnet_g", False, "if use resnet for G")
flags.DEFINE_boolean("if_resnet_d", False, "if use resnet for origin D")
flags.DEFINE_float("lambda_gp", 10.0, "if 'gpwgan' is chosen the corresponding lambda must be filled")
flags.DEFINE_float("clamp_lower", -0.01, "if 'wgan' is chosen the corresponding lambda must be filled, the upper bound of parameters in disc")
flags.DEFINE_float("clamp_upper", 0.01, "if 'wgan' is chosen the corresponding lambda must be filled, the upper bound of parameters in disc")

flags.DEFINE_string("E_norm", "instance", "normalization options:[instance, batch, norm]")
flags.DEFINE_string("G_norm", "instance", "normalization options:[instance, batch, norm]")
flags.DEFINE_string("D_norm", "instance", "normalization options:[instance, batch, norm]")
flags.DEFINE_string("D_patch_norm", "batch", "normalization options:[instance, batch, norm]")
flags.DEFINE_boolean("use_D_origin", True, "True for using origin discriminator")
flags.DEFINE_string("originD_inputForm", "concat_w", "concat_w, concat_n")

flags.DEFINE_boolean("use_D_patch", False, "True for using patch discriminator, modify the network setting")

flags.DEFINE_boolean("use_D_patch2", True, "True for using patch discriminator, modify the size of input of discriminator")
# flags.DEFINE_integer("scale_num", 2, "num of of multi-discriminator")
flags.DEFINE_integer("sizeOfIn_patch2", 128, "The size of input for D_patch2")
flags.DEFINE_string("conditional_D2", "single_right", "full_concat_w, full_concat_n, single_right")

flags.DEFINE_boolean("use_D_patch2_2", False, "True for using patch discriminator, modify the size of input of discriminator")
# flags.DEFINE_integer("scale_num", 2, "num of of multi-discriminator")
flags.DEFINE_integer("sizeOfIn_patch2_2", 256, "The size of input for D_patch2_2")

flags.DEFINE_boolean("use_D_patch3", True, "True for using patch discriminator, modify the size of input of discriminator, user for edge discriminator when G_num == 2")
flags.DEFINE_integer("sizeOfIn_patch3", 128, "The size of input for D_patch2")
flags.DEFINE_string("conditional_D3", "single_right", "full_concat_n, full_concat_w, single_right")

flags.DEFINE_boolean("use_patchGAN_D_full", False, "True for using patchGAN D")
flags.DEFINE_string("patchGAN_D_norm", "instance", "normalization options:[instance, batch, norm]")
flags.DEFINE_string("patchGAN_loss", "origin", "[wgan, gpwgan, origin]")

flags.DEFINE_float("D_origin_loss", 1.0, "weight of origin discriminative loss, is ineffective when use_D_origin is false")
flags.DEFINE_float("D_patch_loss", 0.0, "weight of patch discriminative loss, is ineffective when use_D_patch is false")
flags.DEFINE_float("D_patch2_loss", 1.0, "weight of patch discriminative loss, is ineffective when use_D_patch2 is false")
flags.DEFINE_float("D_patch2_2_loss", 0.5, "weight of patch discriminative loss, is ineffective when use_D_patch2_2 is false")
flags.DEFINE_float("D_patch3_loss", 1.0, "weight of patch discriminative loss, is ineffective when use_D_patch3 is false")
flags.DEFINE_float("D_patchGAN_loss", 1.0, "weight of patch discriminative loss, is ineffective when use_D_patchGAN_D_full is false")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    dir_tmp = ''
    dir_tmp += 'originD'
    dir_tmp += "ConcatW_"
    if FLAGS.use_D_patch2 == True:
        dir_tmp += 'patch2_'
        dir_tmp += str(FLAGS.sizeOfIn_patch2)
        dir_tmp += '_'
    if FLAGS.use_D_patch3 == True:
        dir_tmp += 'patch3_'
        dir_tmp += str(FLAGS.sizeOfIn_patch3)
        dir_tmp += '_'
    dir_tmp += '2G'
    FLAGS.checkpoint_dir = 'checkpoint_' + FLAGS.type + '_instanceEGD' + '_' + dir_tmp
    FLAGS.sample_dir = 'samples_' + FLAGS.type + '_instanceEGD' + '_' + dir_tmp

    print(FLAGS.checkpoint_dir)
    print(FLAGS.sample_dir)

    makedirs(FLAGS.checkpoint_dir)
    makedirs(FLAGS.checkpoint_dir+"/stage1")
    makedirs(FLAGS.checkpoint_dir+"/stage2")
    makedirs(FLAGS.checkpoint_dir + "/stage1_AddE/")
    makedirs(FLAGS.sample_dir)
    makedirs(FLAGS.sample_dir+"/stage1")
    makedirs(FLAGS.sample_dir+"/stage2")
    makedirs(FLAGS.sample_dir + "/stage1_AddE_specified/")
    makedirs(FLAGS.sample_dir + "/stage1_AddE_random/" + FLAGS.dataset + '/')
    makedirs(FLAGS.sample_dir + "/stage1_AddE_specified/" + FLAGS.dataset + '/' + str(FLAGS.test_label) + '/')

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(sess, FLAGS)

        if FLAGS.train:
            dcgan.train1()
        else:
            if FLAGS.Test_allLabel:
                for label in xrange(0, FLAGS.num_classes):
                    FLAGS.test_label = label
                    if FLAGS.output_form is "batch":
                        makedirs(FLAGS.sample_dir + "/stage1_AddE_specified/" + FLAGS.dataset + '/' + str(
                            FLAGS.test_label) + '/')
                    else:
                        makedirs(FLAGS.sample_dir + "/stage1_AddE_specified/" + FLAGS.dataset + '_singleTest/' + str(
                            FLAGS.test_label) + '/')
                    dcgan.test2()
            elif not FLAGS.Random_test:
                if FLAGS.output_form is "batch":
                    makedirs(FLAGS.sample_dir + "/stage1_AddE_specified/" + FLAGS.dataset + '/' + str(
                        FLAGS.test_label) + '/')
                else:
                    makedirs(FLAGS.sample_dir + "/stage1_AddE_specified/" + FLAGS.dataset + '_singleTest/' + str(
                        FLAGS.test_label) + '/')
                dcgan.test2()
            elif FLAGS.Random_test and FLAGS.Test_singleLabel:
                dcgan.test1(FLAGS.num_classes * FLAGS.batch_size)
            else:
                dcgan.test1(10*FLAGS.batch_size)



if __name__ == '__main__':
    tf.app.run()
