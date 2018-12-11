import argparse
import sys

parser = argparse.ArgumentParser(description='manual to this script')

parser.add_argument('--model_save_path', type=str, default=sys.path[0] + '/model/',
                    help='The path where model shall be saved')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size during training')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Epochs during training')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learing rate')
parser.add_argument('--corpus', type=str, default=sys.path[0] + '/xiaohuangji50w_nofenci.conv',
                    help='The corpus file path')
parser.add_argument('--most_qa', type=int, default=10000,
                    help='The max length of QA dictionary')
parser.add_argument('--embedding_qa_size', type=int, default=300,
                    help='Embedding size for QA words')
parser.add_argument('--hidden_units', type=int, default=200,
                    help='Hidden units for module')
parser.add_argument('--n_layers', type=int, default=2,
                    help='number of GRU layers')
parser.add_argument('--drop_prob', type=float, default=0.5,
                    help='The probility used to dropout of output of LSTM or GRU')
parser.add_argument('--mode', type=str, default='train0',
                    help='The mode of train or predict as follows: '
                         'train0: train first time or retrain'
                         'train1: continue train'
                         'predict: predict')
parser.add_argument('--per_save', type=int, default=10,
                    help='save model for every per_save')

params = parser.parse_args()
