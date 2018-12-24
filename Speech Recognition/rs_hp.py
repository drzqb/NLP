import argparse

parser = argparse.ArgumentParser(description='manual description to speech recognition')

parser.add_argument('--epochs', type=int, default=500, help='epochs of iteration')
parser.add_argument('--batch_size', type=int, default=128, help='epochs of iteration')
parser.add_argument('--n_hidden_units', type=int, default=128, help='hidden units for GRU or LSTM')
parser.add_argument('--n_classes', type=int, default=10, help='classification of sound')
parser.add_argument('--n_layers', type=int, default=2, help='number of layers for GRU or LSTM')
parser.add_argument('--p_keep', type=float, default=0.5, help='probility of dropout for GRU or LSTM')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--mode', type=str, default='train0', help='train0 or train1 or predict')
parser.add_argument('--per_save', type=int, default=10, help='save model per per_save')

CONFIG = parser.parse_args()
