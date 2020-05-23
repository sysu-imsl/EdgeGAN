import numpy as np
import tensorflow as tf


def get_acgan_loss_focal(real_image_logits_out, real_image_label,
                         disc_image_logits_out, condition,
                         num_classes, ld1=1, ld2=0.5, ld_focal=2.):
    loss_ac_d = tf.reduce_mean((1 - tf.reduce_sum(tf.nn.softmax(real_image_logits_out) * tf.squeeze(
        tf.one_hot(real_image_label, num_classes, on_value=1., off_value=0., dtype=tf.float32)), axis=1)) ** ld_focal *
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_image_logits_out, labels=real_image_label))
    loss_ac_d = ld1 * loss_ac_d

    loss_ac_g = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_image_logits_out, labels=condition))
    loss_ac_g = ld2 * loss_ac_g
    return loss_ac_g, loss_ac_d


def get_class_loss(logits_out, label, num_classes, ld_focal=2.0):
    loss = tf.reduce_mean((1 - tf.reduce_sum(tf.nn.softmax(logits_out) * tf.squeeze(
        tf.one_hot(label, num_classes, on_value=1., off_value=0., dtype=tf.float32)), axis=1)) ** ld_focal *
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=label))
    return loss


def gradient_penalty(output, on):
    gradients = tf.gradients(output, [on, ])[0]
    grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    return tf.reduce_mean((grad_l2-1)**2)


def discriminator_ganloss(output, target):
    return tf.reduce_mean(output - target)


def generator_ganloss(output):
    return tf.reduce_mean(output * -1)


def l1loss(output, target, weight):
    return weight * tf.reduce_mean(tf.abs(output - target))


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])
