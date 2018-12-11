'''
    seq2seq with attention after gru layers
             and inference
             and two gru layers
             for neural translation
             using pytorch
'''
import numpy as np
import torch
from torch.nn import Embedding, Linear, GRU, CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import sys, collections, re, os
import jieba, zhon.hanzi, string
import matplotlib.pylab as plt
from nltk.tokenize import WordPunctTokenizer
from nmt_hp import FLAGS


class Lang():
    '''
    读取中英文语料并建立中英文字典
    '''

    def __init__(self):
        if isinstance(FLAGS.corpus, list) or isinstance(FLAGS.corpus, tuple):
            self.read_raw_data()
        else:
            self.read_data()

        self.create_dict()
        self.text2id()

    # 读取中英文语料
    def read_data(self):
        self.raw_en = []
        self.raw_ch = []
        s = open(FLAGS.corpus, 'r', encoding='utf-8').read()
        s = s.split('\n')

        for sl in s[:]:
            tmp_sl = sl.strip(' ')
            tmp_sl = re.sub(' ', '', tmp_sl)
            tmp_sl = re.sub('\t', '', tmp_sl)
            if tmp_sl == '':
                s.remove(sl)

        for sl in s[1:len(s):2]:
            self.raw_en.append(WordPunctTokenizer().tokenize(sl.lower().strip(' ')))

        for sl in s[0:-1:2]:
            self.raw_ch.append(jieba.lcut(re.sub(' |\u3000', '', sl.strip(' '))))

        tmp_en = self.raw_en.copy()
        tmp_ch = self.raw_ch.copy()
        for i in reversed(range(len(tmp_en))):
            if len(tmp_en[i]) > 50:
                self.raw_en.pop(i)
                self.raw_ch.pop(i)

    # 读取中英文语料
    def read_raw_data(self):
        self.raw_en = []
        self.raw_ch = []
        with open(FLAGS.corpus[0], 'r') as f:
            for line in f.readlines():
                self.raw_en.append(line.lower().strip('\n').split())

        with open(FLAGS.corpus[1], 'r') as f:
            for line in f.readlines():
                self.raw_ch.append(line.strip('\n').split())

    # 建立中英文字典
    def create_dict(self):
        self.en_dict = dict()
        self.ch_dict = dict()

        tmp_raw_en = []
        for en in self.raw_en:
            tmp_raw_en.extend(en)
        counter = collections.Counter(tmp_raw_en).most_common(FLAGS.most_en - 2)

        self.en_dict['<PAD>'] = len(self.en_dict)
        self.en_dict['<UNK>'] = len(self.en_dict)
        for word, _ in counter:
            self.en_dict[word] = len(self.en_dict)

        self.en_reverse_dict = {v: k for k, v in self.en_dict.items()}

        tmp_raw_ch = []
        for ch in self.raw_ch:
            tmp_raw_ch.extend(ch)
        counter = collections.Counter(tmp_raw_ch).most_common(FLAGS.most_ch - 4)

        self.ch_dict['<PAD>'] = len(self.ch_dict)
        self.ch_dict['<EOS>'] = len(self.ch_dict)
        self.ch_dict['<UNK>'] = len(self.ch_dict)
        self.ch_dict['<GO>'] = len(self.ch_dict)
        for word, _ in counter:
            self.ch_dict[word] = len(self.ch_dict)

        self.ch_reverse_dict = {v: k for k, v in self.ch_dict.items()}

    # 语料向量化
    def text2id(self):
        self.entext2id = []
        self.entext2id_length = []
        self.chtext2id_target = []
        self.chtext2id_input = []
        self.chtext2id_input_length = []

        for en in self.raw_en:
            tmp = []
            for word in en:
                tmp.append(self.en_dict[word] if word in self.en_dict.keys() else self.en_dict['<UNK>'])
            self.entext2id.append(tmp)
            self.entext2id_length.append(len(tmp))

        for ch in self.raw_ch:
            tmp = []
            for word in ch:
                tmp.append(self.ch_dict[word] if word in self.ch_dict.keys() else self.ch_dict['<UNK>'])

            tmp1 = tmp.copy()
            tmp1.insert(0, self.ch_dict['<GO>'])
            self.chtext2id_input.append(tmp1)
            self.chtext2id_input_length.append(len(tmp1))

            tmp.append(self.ch_dict['<EOS>'])
            self.chtext2id_target.append(tmp)


class Encoder(torch.nn.Module):
    def __init__(self, l_dict_en=1000):
        super(Encoder, self).__init__()
        self.l_dict_en = l_dict_en
        self.encoder_embedding = Embedding(l_dict_en, FLAGS.embedding_en_size)
        self.encoder_gru = GRU(input_size=FLAGS.embedding_en_size, hidden_size=FLAGS.hidden_units,
                               num_layers=FLAGS.n_layers, dropout=FLAGS.drop_prob, batch_first=True)

    def forward(self, encoder_input, encoder_input_length, hidden):
        # encoder_input: B*L
        # hidden: n_layers*B*hidden_units
        if encoder_input.size(0) > 1:
            encoder_input = pack_padded_sequence(Variable(self.encoder_embedding(encoder_input)),
                                                 encoder_input_length, batch_first=True)
            output, state = self.encoder_gru(encoder_input, hidden)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, state = self.encoder_gru(self.encoder_embedding(encoder_input), hidden)

        # output: B*L*hidden_units
        # state: n_layers*B*hidden_units
        return output, state

    def initial_hidden(self, batch_size):
        return torch.zeros(FLAGS.n_layers, batch_size, FLAGS.hidden_units)


class AttnDecoder(torch.nn.Module):
    def __init__(self, l_dict_ch):
        super(AttnDecoder, self).__init__()
        self.l_dict_ch = l_dict_ch

        self.decoder_embedding = Embedding(l_dict_ch, FLAGS.embedding_ch_size)
        self.decoder_gru = GRU(input_size=FLAGS.embedding_ch_size,
                               hidden_size=FLAGS.hidden_units,
                               num_layers=FLAGS.n_layers,
                               dropout=FLAGS.drop_prob,
                               batch_first=True)

        self.attn = Linear(FLAGS.hidden_units, FLAGS.hidden_units, bias=False)
        self.project = Linear(2*FLAGS.hidden_units, l_dict_ch)

    def forward(self, input, hidden, encoder_hiddens):
        # input: B(=1)*L(=1)
        # hidden: n_layers*B(=1)*hidden_units
        # encoder_hiddens: B(=1)*encoder_L*hidden_units
        output, hidden = self.decoder_gru(self.decoder_embedding(input), hidden)
        encoder_hiddens = torch.squeeze(encoder_hiddens, 0)
        ei = self.attn(torch.squeeze(output,1))
        eij = F.softmax(ei.mm(encoder_hiddens.transpose(1, 0)), dim=1)
        ci = eij.mm(encoder_hiddens).view(1, 1, -1)
        inputs = torch.cat([ci, output], dim=2)
        output = self.project(inputs)

        return output, hidden


class Usr():
    def train(self, entext2id, entext2id_length, en_dict,
              chtext2id_input, chtext2id_input_length, ch_dict, ch_reverse_dict, chtext2id_target):
        encoder = Encoder(len(en_dict))
        attndecoder = AttnDecoder(len(ch_dict))

        if FLAGS.mode == 'train1':
            encoder.load_state_dict(torch.load(FLAGS.model_save_path + 'nmt4_encoder.pkl'))
            attndecoder.load_state_dict(torch.load(FLAGS.model_save_path + 'nmt4_attndecoder.pkl'))

        loss = CrossEntropyLoss()
        params = list(encoder.parameters()) + list(attndecoder.parameters())
        optimizer = torch.optim.Adam(params, lr=FLAGS.lr)

        test = ['In old China , the people did not have human rights .',
                'The physical conditions of women have greatly improved .',
                'In old china there was not even the most basic medical and health service for ordinary people .',
                'China has cracked down on serious criminal offences in accordance with law .',
                'Children are the future and hope of mankind .']

        test_strip = [test[i].lower().split() for i in range(len(test))]
        test2id = []
        for i in range(len(test)):
            tmp = []
            for word in test_strip[i]:
                tmp.append(en_dict[word] if word in en_dict.keys() else en_dict['<UNK>'])
            test2id.append(tmp)

        test2id_len = [len(test2id[i]) for i in range(len(test))]
        test_batch = test2id
        test_batch_len = test2id_len

        print(test_batch)
        print(test_batch_len)

        test_result_batch_len = [15, 20, 30, 20, 10]

        idx = np.argsort(-np.array(test_batch_len))

        test = self.rearrange(test, idx)
        x_test_input_batch = self.rearrange(self.padding(test_batch, test_batch_len, en_dict['<PAD>']), idx)
        x_test_input_batch_length = self.rearrange(test_batch_len, idx)
        y_test_input_batch_length = self.rearrange(test_result_batch_len, idx)

        m_samples = len(entext2id)
        total_batch = m_samples // FLAGS.batch_size

        loss_ = []
        for epoch in range(1, FLAGS.epochs + 1):
            loss_epoch = 0.0
            for batch in range(total_batch):
                encoder.train()
                attndecoder.train()

                loss_batch = 0.0

                x_input_batch = entext2id[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
                y_input_batch = chtext2id_input[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
                y_target_batch = chtext2id_target[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]

                x_input_batch_length = entext2id_length[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]
                y_input_batch_length = chtext2id_input_length[batch * FLAGS.batch_size: (batch + 1) * FLAGS.batch_size]

                idx = np.argsort(-np.array(x_input_batch_length))
                x_input_batch = self.rearrange(self.padding(x_input_batch, x_input_batch_length, en_dict['<PAD>']), idx)
                y_input_batch = self.rearrange(self.padding(y_input_batch, y_input_batch_length, ch_dict['<PAD>']), idx)
                y_target_batch = self.rearrange(self.padding(y_target_batch, y_input_batch_length, ch_dict['<PAD>']), idx)

                x_input_batch_length = self.rearrange(x_input_batch_length, idx)
                y_input_batch_length = self.rearrange(y_input_batch_length, idx)

                encoder_initial_state = encoder.initial_hidden(FLAGS.batch_size)
                encoder_output, final_state = encoder(torch.from_numpy(np.array(x_input_batch, dtype=np.int64)),
                                                      x_input_batch_length, encoder_initial_state)
                for i in range(FLAGS.batch_size):
                    loss_time = 0.0
                    decoder_state = final_state[:, i:i + 1]
                    H = encoder_output[i:i + 1, :x_input_batch_length[i]]

                    for j in range(y_input_batch_length[i]):
                        decoder_output, decoder_state = attndecoder(
                            torch.from_numpy(np.array(y_input_batch[i:i + 1], dtype=np.int64)[:, j:j + 1]),
                            decoder_state,
                            H)
                        loss_time += loss(decoder_output.squeeze(1),
                                          torch.from_numpy(np.array(y_target_batch[i][j:j + 1])))
                    loss_batch += loss_time / y_input_batch_length[i]
                loss_batch /= FLAGS.batch_size

                loss_epoch += loss_batch.data.numpy()

                encoder.zero_grad()
                attndecoder.zero_grad()
                loss_batch.backward()
                torch.nn.utils.clip_grad_value_(params, 5)
                optimizer.step()

                encoder.train(False)
                attndecoder.train(False)

                sys.stdout.write('>> %d/%d | %d/%d\n' % (epoch, FLAGS.epochs, batch + 1, total_batch))

                encoder_initial_state = encoder.initial_hidden(len(test))
                encoder_output, final_state = encoder(torch.from_numpy(np.array(x_test_input_batch, dtype=np.int64)),
                                                      x_test_input_batch_length,
                                                      encoder_initial_state)

                for i in range(len(test)):
                    sys.stdout.write('English: %s\n' % (test[i]))
                    result = []
                    decoder_input = torch.from_numpy(ch_dict['<GO>'] * np.ones([1, 1], dtype=np.int64))
                    decoder_state = final_state[:, i:i + 1]
                    H = encoder_output[i:i + 1, :x_test_input_batch_length[i]]
                    for j in range(y_test_input_batch_length[i]):
                        decoder_output, decoder_state = attndecoder(decoder_input,
                                                                    decoder_state,
                                                                    H)
                        decoder_output = torch.argmax(decoder_output, dim=2)
                        check_eos = decoder_output.numpy()[0, 0]
                        if check_eos == ch_dict['<EOS>']:
                            break
                        else:
                            result.append(ch_reverse_dict[check_eos])
                            decoder_input = decoder_output
                    sys.stdout.write('Chinese: %s\n\n' % (''.join(result)))
                sys.stdout.flush()

            loss_epoch /= total_batch
            loss_.append(loss_epoch)

            print('\033[1;31;40m')
            print('>> %d/%d | Loss:%.9f' % (epoch, FLAGS.epochs, loss_[-1]))
            print('\033[0m')

            r = np.random.permutation(m_samples)
            entext2id = self.rearrange(entext2id, r)
            entext2id_length = self.rearrange(entext2id_length, r)
            chtext2id_input = self.rearrange(chtext2id_input, r)
            chtext2id_input_length = self.rearrange(chtext2id_input_length, r)
            chtext2id_target = self.rearrange(chtext2id_target, r)

            if epoch % FLAGS.per_save == 0:
                torch.save(encoder.state_dict(), FLAGS.model_save_path + 'nmt4_encoder.pkl')
                torch.save(attndecoder.state_dict(), FLAGS.model_save_path + 'nmt4_attndecoder.pkl')

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(FLAGS.model_save_path + 'nmt4_loss.png')
        plt.close(fig)

    def predict(self, en_dict, ch_dict, ch_reverse_dict):
        encoder = Encoder(len(en_dict))
        attndecoder = AttnDecoder(len(ch_dict))

        encoder.load_state_dict(torch.load(FLAGS.model_save_path + 'nmt4_encoder.pkl'))
        attndecoder.load_state_dict(torch.load(FLAGS.model_save_path + 'nmt4_attndecoder.pkl'))
        encoder.train(False)
        attndecoder.train(False)
        print('\033[1;31;40m')

        while 1:
            test = input('Please enter english sentence or q to quit\n')
            if test.lower()=='q':
                break
            test_strip = [WordPunctTokenizer().tokenize(test.lower().strip(' '))]
            test_len = [len(test_strip[0])]
            test2id = []
            tmp = []
            for word in test_strip[0]:
                tmp.append(en_dict[word] if word in en_dict.keys() else en_dict['<UNK>'])
            test2id.append(tmp)

            x_test_input_batch = test2id
            x_test_input_batch_length = test_len
            y_test_input_batch_length = [30]

            encoder_initial_state = encoder.initial_hidden(1)
            encoder_output, final_state = encoder(torch.from_numpy(np.array(x_test_input_batch, dtype=np.int64)),
                                                  x_test_input_batch_length,
                                                  encoder_initial_state)

            result = []
            decoder_input = torch.from_numpy(ch_dict['<GO>'] * np.ones([1, 1], dtype=np.int64))
            decoder_state = final_state[:, i:i + 1]
            H = encoder_output[0:1, :x_test_input_batch_length[0]]
            for j in range(y_test_input_batch_length[0]):
                decoder_output, decoder_state = attndecoder(decoder_input,
                                                            decoder_state,
                                                            H)
                decoder_output = torch.argmax(decoder_output, dim=2)
                check_eos = decoder_output.numpy()[0, 0]
                if check_eos == ch_dict['<EOS>']:
                    break
                else:
                    result.append(ch_reverse_dict[check_eos])
                    decoder_input = decoder_output
            sys.stdout.write('Chinese: %s\n\n' % (''.join(result)))
            sys.stdout.flush()
        print('\033[0m')

    def padding(self, x, l, padding_id):
        l_max = np.max(l)
        return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]

    def rearrange(self, x, r):
        return [x[ri] for ri in r]


def main():
    if not os.path.exists(FLAGS.model_save_path):
        os.makedirs(FLAGS.model_save_path)

    lang = Lang()
    print(len(lang.raw_en), lang.raw_en[:10])
    print(len(lang.raw_ch), lang.raw_ch[:10])

    print(len(lang.en_dict), [lang.en_reverse_dict[i] for i in range(100)])
    print(len(lang.ch_dict), [lang.ch_reverse_dict[i] for i in range(100)])

    print(lang.entext2id_length[:10], np.max(lang.entext2id_length), np.min(lang.entext2id_length),
          np.argmin(lang.entext2id_length), lang.entext2id[:10])
    print(lang.chtext2id_input_length[:10], np.max(lang.chtext2id_input_length), np.min(lang.chtext2id_input_length),
          np.argmin(lang.chtext2id_input_length), lang.chtext2id_target[:10])
    print(lang.chtext2id_input_length[:10], np.max(lang.chtext2id_input_length), np.min(lang.chtext2id_input_length),
          np.argmin(lang.chtext2id_input_length), lang.chtext2id_input[:10])

    f = open(sys.path[0] + '/bps2.txt', 'w+')
    for i in range(len(lang.entext2id)):
        f.write(' '.join(lang.raw_ch[i]) + '\n')
        f.write(' '.join(lang.raw_en[i]) + '\n')
        f.write('\n')
    f.close()
    # exit()

    usr = Usr()
    if FLAGS.mode == 'predict':
        usr.predict(lang.en_dict, lang.ch_dict, lang.ch_reverse_dict)
    else:
        usr.train(lang.entext2id, lang.entext2id_length, lang.en_dict,
                  lang.chtext2id_input, lang.chtext2id_input_length,
                  lang.ch_dict, lang.ch_reverse_dict,
                  lang.chtext2id_target)


if __name__ == '__main__':
    main()
