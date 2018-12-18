import tensorflow as tf

tf.flags.DEFINE_string('corpus', 'text8.zip', 'filename of corpus')
tf.flags.DEFINE_integer('vocab_size', 10000, 'size of word dict')
tf.flags.DEFINE_integer('num_samples', 64, 'number of samples in nce_loss')
tf.flags.DEFINE_integer('embed_size', 256, 'size of dense representation of word')
tf.flags.DEFINE_integer('window_size', 1, 'size of slide window in cbow model')
tf.flags.DEFINE_integer('epochs', 10000, 'number of iteration')
tf.flags.DEFINE_integer('batch_size', 1024, 'batch size')
tf.flags.DEFINE_integer('plot_total', 1000, 'number of words plotted in 2D figure')
tf.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.flags.DEFINE_string('cbow_save', 'cbow.png', 'filename of saved word vectors for cbow')
tf.flags.DEFINE_string('loss_cbow_save', 'loss_cbow.png', 'picture of saved loss for cbow')
tf.flags.DEFINE_string('skipgram_save', 'skipgram.png', 'filename of saved word vectors for skipgram')
tf.flags.DEFINE_string('loss_skipgram_save', 'loss_skipgram.png', 'picture of saved loss for skipgram')
FLAGS = tf.flags.FLAGS
