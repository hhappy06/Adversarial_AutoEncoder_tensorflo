import os, sys
import tensorflow as tf
import numpy as np
import time
from model.aae import AAE
from input.line_parser.line_parser import ImageParser
from input.data_reader import read_data
from loss.aae_loss import AAELoss
from train_op import autoencoder_train_opt, adversarial_train_opt, generator_train_opt
from util import util

_BATCH_SIZE_ = 64
_IMAGE_SIZE_ = 28
_Z_DIM_ = 2
_N_LABEL = 10
_EPOCH_ = 1
_TRAINING_SET_SIZE_ = 60000
_DATA_DIR_ = './data/mnist/train_images'
_CSVFILE_ = ['./data/mnist/train_images/file_list']

_OUTPUT_INFO_FREQUENCE_ = 100
_OUTPUT_IMAGE_FREQUENCE_ = 100

_TESTING_SAMPEL_NUMBER_ = 2
_TESTING_OUTPUT_FILE_ = './result/test.res'

line_parser = ImageParser()
aae_loss = AAELoss()

def train():
	with tf.Graph().as_default():
		images, image_labels = read_data(_CSVFILE_, line_parser = line_parser, data_dir = _DATA_DIR_, batch_size = _BATCH_SIZE_)
		image_one_hot_lables = util.one_hot_label_tensor(_N_LABEL, image_labels)
		z = tf.placeholder(tf.float32, [None, _Z_DIM_], name = 'z')
		one_hot_z = tf.placeholder(tf.float32, [None, _N_LABEL], name = 'one_hot_z')

		aae = AAE()
		d_output_z, d_output_images_z, decoder_images, encode_images = aae.inference(images, _IMAGE_SIZE_, image_one_hot_lables, z, one_hot_z)
		adv_loss, gen_loss, decode_loss = aae_loss.loss(d_output_z, d_output_images_z, decoder_images, images)

		# opt
		trainable_vars = tf.trainable_variables()
		auto_vars = [var for var in trainable_vars if 'decoder' in var.name]
		adver_vars = [var for var in trainable_vars if 'discriminator' in var.name]
		gen_vars = [var for var in trainable_vars if 'encoder' in var.name]

		auto_opt = autoencoder_train_opt(decode_loss, auto_vars)
		adver_opt = adversarial_train_opt(adv_loss, adver_vars)
		gen_opt = generator_train_opt(gen_loss, gen_vars)

		# generate_images for showing
		generate_images = aae.generate_images(z, _IMAGE_SIZE_, 4, 4)
		
		# summary
		sum_z = tf.summary.histogram('z', z)
		sum_decode_loss = tf.summary.scalar('decode_loss', decode_loss)
		sum_adv_loss = tf.summary.scalar('adv_loss', adv_loss)
		sum_gen_loss = tf.summary.scalar('gen_loss', gen_loss)

		sum_auto = tf.summary.merge([sum_decode_loss])
		sum_adv = tf.summary.merge([sum_z, sum_adv_loss])
		sum_gen = tf.summary.merge([sum_gen_loss])

		# initialize variable
		init_op = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.global_variables())

		session = tf.Session()
		file_writer = tf.summary.FileWriter('./logs', session.graph)
		session.run(init_op)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		print 'AAE training starts...'
		sys.stdout.flush()
		counter = 0
		max_steps = int(_TRAINING_SET_SIZE_ / _BATCH_SIZE_)
		for epoch in xrange(_EPOCH_):
			for step in xrange(max_steps):
				batch_z_label = np.random.randint(low = 0, high = _N_LABEL, size = _BATCH_SIZE_)
				batch_z, batch_z_one_hot_label = util.sample_z_from_n_2d_gaussian_mixture(len(batch_z_label), batch_z_label, _N_LABEL)

				_, summary_str, error_decode_loss = session.run([auto_opt, sum_auto, decode_loss])
				file_writer.add_summary(summary_str, counter)

				_, summary_str, error_adv_loss = session.run([adver_opt, sum_adv, adv_loss], feed_dict = {
					z: batch_z,
					one_hot_z: batch_z_one_hot_label})
				file_writer.add_summary(summary_str, counter)

				_, summary_str, error_gen_loss = session.run([gen_opt, sum_gen, gen_loss], feed_dict = {
					z: batch_z,
					one_hot_z: batch_z_one_hot_label})
				file_writer.add_summary(summary_str, counter)

				_, summary_str, error_gen_loss = session.run([gen_opt, sum_gen, gen_loss], feed_dict = {
					z: batch_z,
					one_hot_z: batch_z_one_hot_label})
				file_writer.add_summary(summary_str, counter)

				file_writer.flush()

				counter += 1

				if counter % _OUTPUT_INFO_FREQUENCE_ == 0:
					print 'step: (%d, %d), adver_loss: %f, gen_loss: %f, auto_loss:%f'%(epoch, step, error_adv_loss, error_gen_loss, error_decode_loss)
					sys.stdout.flush()

				if counter % _OUTPUT_IMAGE_FREQUENCE_ == 0:
					batch_z = np.random.uniform(-1, 1, [_BATCH_SIZE_, _Z_DIM_]).astype(np.float32)
					batch_z, batch_z_one_hot_label = util.sample_z_from_n_2d_gaussian_mixture(len(batch_z_label), batch_z_label, _N_LABEL)
					generated_image_eval = session.run(generate_images, {z: batch_z})
					filename = os.path.join('./result', 'out_%03d_%05d.png' %(epoch, step))
					with open(filename, 'wb') as f:
						f.write(generated_image_eval)
					print 'output generated image: %s'%(filename)
					sys.stdout.flush()

		print 'training done!'
		file_writer.close()

		# testing the result
		with open(_TESTING_OUTPUT_FILE_, 'w') as output_file:
			print 'testing output file...'
			for setp in xrange(_TESTING_SAMPEL_NUMBER_):
				test_encode, test_label = session.run([encode_images, image_labels])
				test_str = ''.join(['%f,%f,%d\n'%(data[0][0], data[0][1], data[1]) for data in zip(test_encode, test_label)])
				output_file.write(test_str)

		coord.request_stop()
		coord.join(threads)
		session.close()

if __name__ == '__main__':
	train()
