from __future__ import print_function
from six.moves import xrange
import tensorflow.contrib.slim as slim
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from functools import partial
from tensorflow.contrib import rnn
import random
import collections
import time

batch_size = 32
z_dim = 128
learning_rate_ger = 5e-5
learning_rate_dis = 5e-5

# numbers of words in you generated paragraph
sent_len = 50
n_hidden = 128

# Import data for training discriminator
step = 1
start_time = time.time()
iters = 500000
Citer = 5
log_dir = './log_wgan'
ckpt_dir = './ckpt_wgan'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
lam = 10.

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

training_file = '0.txt'
training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# Generator use Encoder-Decoder model
def generator_autoencoder(z):
	z = tf.reshape(z,[batch_size, sent_len * vocab_size])
	train = ly.fully_connected(
		z, sent_len * vocab_size, activation_fn=lrelu, normalizer_fn=ly.batch_norm,
		normalizer_params={'is_training':True})
	train = ly.fully_connected(
		train, sent_len * vocab_size, activation_fn=None, normalizer_fn=ly.batch_norm,
		normalizer_params={'is_training':True})
	# Output use Softmax for each vector, here we maintain linear activation
	train = tf.reshape(train, [batch_size, sent_len, vocab_size])
	train = tf.nn.softmax(train)
	return train

# Discriminator use LSTM model
def discriminator_LSTM(paragraph, reuse=False):
	with tf.variable_scope('critic') as scope:
		if reuse:
			scope.reuse_variables()
		par = tf.reshape(paragraph, [-1, sent_len, vocab_size])
		par = tf.transpose(par, [1, 0, 2])
		par = tf.reshape(par, [-1, vocab_size])
		par = ly.fully_connected(par, n_hidden, activation_fn=lrelu)
		par = tf.split(par, sent_len, 0)
		lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
		lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
		lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
		outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, par, dtype=tf.float32)
		out = outputs[-1]
		logit = ly.fully_connected(out, 1, activation_fn=None)
	return logit

def build_graph():
	noise_dist = tf.contrib.distributions.Normal(0., 1.)
	z = noise_dist.sample((batch_size, sent_len, vocab_size))
	with tf.variable_scope('generator'):
		train = generator_autoencoder(z)
	real_data = tf.placeholder(
        dtype=tf.float32, shape=(batch_size, sent_len, vocab_size))
	true_logit = discriminator_LSTM(real_data)
	fake_logit = discriminator_LSTM(train, reuse=True)
	c_loss = tf.reduce_mean(fake_logit - true_logit)
	# use WGAN-gp
	alpha_dist = tf.contrib.distributions.Uniform(0., 1.)
	alpha = alpha_dist.sample((batch_size, 1, 1, 1))
	interpolated = real_data + alpha*(train-real_data)
	inte_logit = discriminator_LSTM(interpolated, reuse=True)
	gradients = tf.gradients(inte_logit, [interpolated,])[0]
	grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
	gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
	gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
	grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
	c_loss += lam*gradient_penalty
	#
	g_loss = tf.reduce_mean(-fake_logit)
	g_loss_sum = tf.summary.scalar("g_loss", g_loss)
	c_loss_sum = tf.summary.scalar("c_loss", c_loss)
	theta_g = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
	theta_c = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
	counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	# Not use Adam Optimizer
	is_adam = False
	opt_g = ly.optimize_loss(loss=g_loss, learning_rate=learning_rate_ger,
					optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer, 
					variables=theta_g, global_step=counter_g,
					summaries = ['gradient_norm'])
	counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
	opt_c = ly.optimize_loss(loss=c_loss, learning_rate=learning_rate_dis,
					optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if is_adam is True else tf.train.RMSPropOptimizer, 
					variables=theta_c, global_step=counter_c,
					summaries = ['gradient_norm'])
	return opt_g, opt_c, real_data

def main():
	print('Building Graph...')
	opt_g, opt_c, real_data = build_graph()
	saver = tf.train.Saver()
	merged_all = tf.summary.merge_all()
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.8
	n_input = 50
	trainSet_size = int(len(training_data)/n_input)
	def next_feed_dict():
		global step
		temp = (step - 1) * batch_size
		bind = (temp%trainSet_size)
		eind = (temp+batch_size)%trainSet_size
		if (eind > bind):
			symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(bind * n_input, eind * n_input) ]
		else:
			symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(bind * n_input, len(training_data))]+[ [dictionary[ str(training_data[i])]] for i in(range(eind * n_input))]
		symbols_in_keys = np.reshape(np.array(symbols_in_keys)[:batch_size * n_input], [batch_size, n_input])
		symbols_one_hot = np.eye(vocab_size)[symbols_in_keys]
		feed_dict = {real_data: symbols_one_hot}
		step += 1
		return feed_dict
	with tf.Session(config=config) as sess:
		print('Initializing...')
		sess.run(tf.global_variables_initializer())
		# print('Summary Writer Initing...')
		# summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
		print('Begin Training...')
		for i in range(iters):
			if i < 25 or i % 500 == 0:
				citers = 100
			else:
			    citers = Citers
			print('Training Discriminator:\n')
			for j in range(citers):
				feed_dict = next_feed_dict()
				if i % 100 == 99 and j == 0:
					run_options = tf.RunOptions(
						trace_level=tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					_, merged = sess.run([opt_c, merged_all], feed_dict=feed_dict,
										options=run_options, run_metadata=run_metadata)
					# summary_writer.add_summary(merged, i)
					# summary_writer.add_run_metadata(
					# 	run_metadata, 'critic_metadata {}'.format(i), i)
				else:
					sess.run(opt_c, feed_dict=feed_dict)
			feed_dict = next_feed_dict()
			print('Training Generator:\n')
			if i % 100 == 99:
				_, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,
					options=run_options, run_metadata=run_metadata)
				# summary_writer.add_summary(merged, i)
				# summary_writer.add_run_metadata(
				#	run_metadata, 'generator_metadata {}'.format(i), i)
			else:
				sess.run(opt_g, feed_dict={real_data:symbols_one_hot})          
			if i % 1000 == 999:
				saver.save(sess, os.path.join(ckpt_dir, "model.ckpt"), global_step=i)      

main()









