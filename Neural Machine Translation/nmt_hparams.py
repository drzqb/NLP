import tensorflow as tf
import sys

tf.flags.DEFINE_string('model_save_path', sys.path[0] + '/model/', 'The path where model shall be saved')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch size during training')
tf.flags.DEFINE_integer('epochs', 1000, 'Epochs during training')
tf.flags.DEFINE_float('lr', 0.001, 'Initial learing rate')
tf.flags.DEFINE_string('corpus', sys.path[0] + '/bps1.txt', 'The corpus file path')
tf.flags.DEFINE_integer('most_en', 5000, 'The max length of englishword dictionary')
tf.flags.DEFINE_integer('most_ch', 5000, 'The max length of chineseword dictionary')
tf.flags.DEFINE_integer('embedding_en_size', 200, 'Embedding size for english words')
tf.flags.DEFINE_integer('embedding_ch_size', 200, 'Embedding size for chinese words')
tf.flags.DEFINE_integer('encoder_hidden_units', 500, 'Hidden units for encoder module')
tf.flags.DEFINE_integer('decoder_hidden_units', 500, 'Hidden units for decoder module')
tf.flags.DEFINE_boolean('graph_write', True, 'whether the compute graph is written to logs file')
tf.flags.DEFINE_float('keep_prob', 0.5, 'The probility used to dropout of output of LSTM')
tf.flags.DEFINE_string('mode','train0','The mode of train or predict as follows: '
                                       'train0: train first time or retrain'
                                       'train1: continue train'
                                       'predict: predict')
tf.flags.DEFINE_integer('per_save',10,'save model for every per_save')

FLAGS = tf.flags.FLAGS
