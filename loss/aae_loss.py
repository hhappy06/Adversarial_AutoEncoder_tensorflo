from __future__ import absolute_import
import tensorflow as tf

class AAELoss:
	def loss(self, d_output_z, d_output_image_z, decode_image, ori_image):
		adv_z_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_output_z, tf.ones_like(d_output_z)))
		adv_image_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_output_image_z, tf.zeros_like(d_output_image_z)))
		gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_output_image_z, tf.ones_like(d_output_image_z)))
		decode_loss = tf.reduce_mean(tf.squared_difference(decode_image, ori_image))

		adv_loss = adv_z_loss + adv_image_loss

		return adv_loss, gen_loss, decode_loss