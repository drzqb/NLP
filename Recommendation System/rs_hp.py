import tensorflow as tf

tf.flags.DEFINE_string('datadir', 'ml-1m', 'directory of data files')
tf.flags.DEFINE_integer('user_unit1', 512, 'number of units of first user dense layer')
tf.flags.DEFINE_integer('user_and_movie_unit', 4096, 'number of units of second user and movie dense layer')
tf.flags.DEFINE_integer('movie_unit1', 256, 'number of units of first movie dense layer')
tf.flags.DEFINE_integer('embed_size', 128, 'number of embed size')
tf.flags.DEFINE_integer('gru_unit', 256, 'number of gru hidden units')
tf.flags.DEFINE_integer('epochs', 100, 'number of iteration')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep probility of neural units')
tf.flags.DEFINE_integer('per_save', 10, 'save model per per_save iterations')
tf.flags.DEFINE_string('model_save_path', 'model/', 'save model per per_save iterations')
tf.flags.DEFINE_bool('graph_write', True, 'whether the computing graph is written for save')
tf.flags.DEFINE_string('mode', 'train', 'whether the mode is \'train\' or \'infer\'')
tf.flags.DEFINE_integer('top_k', 20, 'the first top_k movies recommended')

CONFIG = tf.flags.FLAGS
