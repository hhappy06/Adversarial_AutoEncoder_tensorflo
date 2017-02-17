import tensorflow as tf
import numpy as np

_BETA_ = 0.5
_LEARNING_RATE_ = 2.0e-4

def autoencoder_train_opt(decode_loss, autoencoder_vars):
	opt = tf.train.AdamOptimizer(_LEARNING_RATE_, beta1 = _BETA_).minimize(decode_loss, var_list = autoencoder_vars)
	return opt

def adversarial_train_opt(adversarial_loss, adversarial_vars):
	opt = tf.train.AdamOptimizer(_LEARNING_RATE_, beta1 = _BETA_).minimize(adversarial_loss, var_list = adversarial_vars)
	return opt

def generator_train_opt(generator_loss, generator_vars):
	opt = tf.train.AdamOptimizer(_LEARNING_RATE_, beta1 = _BETA_).minimize(generator_loss, var_list = generator_vars)
	return opt