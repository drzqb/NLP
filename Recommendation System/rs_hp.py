import tensorflow as tf

tf.flags.DEFINE_string('datadir', 'ml-1m', 'directory of data files')
tf.flags.DEFINE_integer('user_unit1', 128, 'number of units of first user dense layer')
tf.flags.DEFINE_integer('user_id_embed_size', 32, 'embed size for user id')
tf.flags.DEFINE_integer('user_and_movie_unit', 1024, 'number of units of second user and movie dense layer')
tf.flags.DEFINE_integer('movie_unit1', 128, 'number of units of first movie dense layer')
tf.flags.DEFINE_integer('movie_id_embed_size', 32, 'embed size for movie id')
tf.flags.DEFINE_integer('title_embed_size', 64, 'embed size for title word')
tf.flags.DEFINE_integer('gru_unit', 128, 'number of gru hidden units')
tf.flags.DEFINE_integer('epochs', 100, 'number of iteration')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_float('keep_prob', 0.5, 'keep probility of neural units')
tf.flags.DEFINE_integer('per_save', 10, 'save model per per_save iterations')
tf.flags.DEFINE_string('model_save_path', 'model/', 'save model per per_save iterations')
tf.flags.DEFINE_bool('graph_write', True, 'whether the computing graph is written for save')
tf.flags.DEFINE_string('mode', 'train0', 'whether the mode is \'train0\' for a new training'
                                         'or \'train1\' for continuing training'
                                         'or \'infer\' for infering')
tf.flags.DEFINE_integer('top_k', 20, 'the first top_k movies recommended')

CONFIG = tf.flags.FLAGS
