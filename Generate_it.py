import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly

batch_size = 5
z_dim = 128
s = 32
s2, s4, s8, s16 =\
	int(s / 2), int(s / 4), int(s / 8), int(s / 16)
is_svhn = True
channel = 3 if is_svhn is True else 1
ckpt_dir = './ckpt_wgan'

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

def get_generator():
	z = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim])
	with tf.variable_scope('generator'):
		train = generator(z)
	theta_g = tf.get_collection(
		tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
	return z, train

def generate_from_ckpt():
	with tf.device('/gpu:2'):
		z, train = get_generator()
	config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.6
	if ckpt_dir != None:
		with tf.Session(config=config) as sess:
			saver = tf.train.Saver()
			saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
			batch_z = np.random.normal(0, 1.0, [batch_size, sent_len, vocab_size]) \
				.astype(np.float32)
			rs = train.eval(feed_dict={z:batch_z})
			for i in range(len(rs.shape[0])):
				paragraph = rs[i]
				for j in range(sent_len):
					generated_index = int(tf.argmax(paragraph[j], 1).eval())
					if j==0:
						sentence = "%s" % (reverse_dictionary[generated_index])
					else:
						sentence = "%s %s" % (sentence,reverse_dictionary[generated_index])
				print(sentence)



