from __future__ import absolute_import
import numpy as np
import tensorflow as tf

_HIDDEN_LAYER_DIM_ = 1000
_REGULAR_FACTOR_ = 1.0e-4
_BATCH_SIZE_ = 64

def _construct_full_connection_layer(input, output_dim, stddev = 0.02, name = 'fc_layer'):
	with tf.variable_scope(name):
		init_weight = tf.random_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
		weight = tf.get_variable(
			name = name + '_weight',
			shape = [input.get_shape()[1], output_dim],
			initializer = init_weight,
			regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
		bias = tf.get_variable(
			name = name + '_bias',
			shape = [output_dim],
			initializer = tf.constant_initializer(0.0),
			regularizer = None)
		fc = tf.matmul(input, weight)
		fc = tf.nn.bias_add(fc, bias)
		return fc

# define AAE network
# define disctriminative network
class Discriminative:
	def __init__(self, name = 'discriminator'):
		self.name = name

	def inference(self, input_z, input_label, output_dim = 1, reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			input_layer = tf.concat(1, [input_z, input_label])

			print 'discriminator input:', input_layer.get_shape()

			hidden0 = tf.nn.relu(_construct_full_connection_layer(input_layer, _HIDDEN_LAYER_DIM_, name = 'di_hidden0'))
			hidden1 = tf.nn.relu(_construct_full_connection_layer(hidden0, _HIDDEN_LAYER_DIM_, name = 'di_hidden1'))
			output_layer = _construct_full_connection_layer(hidden1, output_dim, name = 'di_hidden2')

			print 'discriminator ouput :', output_layer.get_shape()

			return output_layer

class Encoder:
	def __init__(self, name = 'encoder'):
		self.name = name

	def inference(self, images, output_dim, reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			print 'encoder input', images.get_shape()

			if len(images.get_shape()) > 2:
				dim = 1
				for d in images.get_shape().as_list()[1:]:
					dim *= d
				images_reshape = tf.reshape(images, [-1, dim])
			else:
				images_reshape = images

			hidden0 = tf.nn.relu(_construct_full_connection_layer(images_reshape, _HIDDEN_LAYER_DIM_, name = 'en_hidden0'))
			hidden1 = tf.nn.relu(_construct_full_connection_layer(hidden0, _HIDDEN_LAYER_DIM_, name = 'en_hidden1'))
			output_layer = _construct_full_connection_layer(hidden1, output_dim, name = 'en_hidden2')
			
			print "encoder output:", output_layer.get_shape()

			return output_layer

class Decoder:
	def __init__(self, name = 'decoder'):
		self.name = name

	def inference(self, input_z, output_image_size, reuse = False):
		with tf.variable_scope(self.name) as scope:
			if reuse:
				scope.reuse_variables()

			print 'decoder input z:', input_z.get_shape()

			hidden0 = tf.nn.relu(_construct_full_connection_layer(input_z, _HIDDEN_LAYER_DIM_, name = 'de_hidden0'))
			hidden1 = tf.nn.relu(_construct_full_connection_layer(hidden0, _HIDDEN_LAYER_DIM_, name = 'de_hidden1'))
			output_layer = tf.nn.tanh(_construct_full_connection_layer(hidden1, output_image_size * output_image_size, name = 'de_hidden2'))

			output_layer_reshape = tf.reshape(output_layer, [-1, output_image_size, output_image_size, 1])
			
			print "decoder output :", output_layer_reshape.get_shape()

			return output_layer_reshape

class AAE:
	def __init__(self, encoder_name = 'encoder', decoder_name = 'decoder', discriminator_name = 'discriminator'):
		self.encoder_name = encoder_name
		self.decoder_name = decoder_name
		self.discriminator_name = discriminator_name

	def inference(self, images, images_size, images_label, z, z_label):
		encode_dim = z.get_shape()[1]
		# generative
		print 'AAE encoder'
		self.encoder = Encoder(name = self.encoder_name)
		self.encoder_z = self.encoder.inference(images, encode_dim)

		# discriminative
		print "="*100
		print 'AAE discriminator'
		self.discriminator = Discriminative(name = self.discriminator_name)
		self.d_output_z = self.discriminator.inference(z, z_label)
		self.d_output_images_z = self.discriminator.inference(self.encoder_z, images_label, reuse = True)

		print "="*100
		print 'AAE decoder'
		self.decoder = Decoder(name = self.decoder_name)
		self.decoder_images = self.decoder.inference(self.encoder_z, images_size)

		return self.d_output_z, self.d_output_images_z, self.decoder_images, self.encoder_z

	def generate_images(self, z, image_size, row=8, col=8):
		images = tf.cast(tf.mul(tf.add(self.decoder.inference(z, image_size, reuse = True), 1.0), 127.5), tf.uint8)
		images = [image for image in tf.split(0, _BATCH_SIZE_, images)]
		rows = []
		for i in range(row):
			rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
		image = tf.concat(1, rows)
		return tf.image.encode_png(tf.squeeze(image, [0]))
