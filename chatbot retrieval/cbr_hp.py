import tensorflow as tf

tf.flags.DEFINE_integer('batch_size', 100, 'batch size for training')
tf.flags.DEFINE_integer('epochs', 100, 'number of iterations')
tf.flags.DEFINE_integer('embedding_dize', 100, 'embedding size for word embedding')
tf.flags.DEFINE_integer('rnn_size', 128, 'units of rnn')
tf.flags.DEFINE_string('model_save_path', 'model/', 'directory of model file saved')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate for training')
tf.flags.DEFINE_float('keep_prob', 0.5, 'rate for dropout')
tf.flags.DEFINE_multi_integer('top_k', [1,2,5], 'top top_k numbers of a vector')
tf.flags.DEFINE_integer('per_save', 10, 'save model once every per_save iterations')
tf.flags.DEFINE_string('mode','train0','The mode of train or predict as follows: '
                                       'train0: train first time or retrain'
                                       'train1: continue train'
                                       'predict: predict')
CONFIG = tf.flags.FLAGS
