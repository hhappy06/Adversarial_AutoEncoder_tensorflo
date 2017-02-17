from __future__ import absolute_import
import numpy as np
import tensorflow as tf

def one_hot_label_tensor(nlabel, labels):
	labels_reshape = tf.reshape(labels, [-1, 1])
	one_hot_label = tf.one_hot(labels_reshape, nlabel, on_value = 1.0, off_value = 0.0, axis = 1, dtype = tf.float32)
	one_hot_label = tf.reshape(one_hot_label, [-1, nlabel])
	return one_hot_label

def sample_z_from_n_2d_gaussian_mixture(batchsize, label_indices, n_labels=10):
	def sample(x, y, label, n_labels):
		shift = 2.0
		r = 2.0 * np.pi / float(n_labels) * float(label)
		new_x = x * np.cos(r) - y * np.sin(r)
		new_y = x * np.sin(r) + y * np.cos(r)
		new_x += shift * np.cos(r)
		new_y += shift * np.sin(r)
		return np.array([new_x, new_y]).reshape((2,))
	x_var = 0.5
	y_var = 0.18
	x = np.random.normal(0, x_var, (batchsize))
	y = np.random.normal(0, y_var, (batchsize))
	z = np.empty((batchsize, 2), dtype=np.float32)
	one_hot_z = np.zeros((batchsize, n_labels))
	for batch in xrange(batchsize):
		z[batch, 0:2] = sample(x[batch], y[batch], label_indices[batch], n_labels)
		one_hot_z[batch, label_indices[batch]] = 1.0

	return z, one_hot_z
