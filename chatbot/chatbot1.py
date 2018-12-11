'''
    seq2seq with attention
             and inference
             and two gru layers
             for chat_bot
             using pytorch
'''
import numpy as np
import torch
from torch.nn import Embedding, Linear, GRU, CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import sys, collections, re, os
import jieba
import matplotlib.pylab as plt
from chatbot_hp import params


class Lang():
    '''
    读取聊天语料并建立字典
    '''

    def __init__(self):
        if isinstance(params.corpus, list) or isinstance(params.corpus, tuple):
            self.read_raw_data()
        else:
            self.read_data()

        self.create_dict()
        self.text2id()

    # 读取中英文语料
    def read_data(self):
        self.raw_q = []
        self.raw_a = []
        s = open(params.corpus, 'r', encoding='utf-8').read()
        s = s.split('\n')

        print('jieba...')
        k = 1
        for sl in s[:]:
            # if k == 20001:
            #     break
            if sl.startswith('M '):
                sl = sl.strip('M ')
                if sl == '':
                    sl = '？？？'
                if k % 2 == 1:
                    self.raw_q.append(jieba.lcut(sl))
                else:
                    self.raw_a.append(jieba.lcut(sl))

                k += 1

        print('removing long sentences......')
        tmp_q = self.raw_q.copy()
        tmp_a = self.raw_a.copy()
        for i in reversed(range(len(tmp_q))):
            if len(tmp_q[i]) > 20 or len(tmp_a[i]) > 20:
                self.raw_q.pop(i)
                self.raw_a.pop(i)

    # 建立字典
    def create_dict(self):
        self.qa_dict = dict()

        tmp_raw_qa = []
        for q in self.raw_q:
            tmp_raw_qa.extend(q)
        for a in self.raw_a:
            tmp_raw_qa.extend(a)
        counter = collections.Counter(tmp_raw_qa).most_common(params.most_qa - 4)

        self.qa_dict['<PAD>'] = len(self.qa_dict)
        self.qa_dict['<EOS>'] = len(self.qa_dict)
        self.qa_dict['<UNK>'] = len(self.qa_dict)
        self.qa_dict['<GO>'] = len(self.qa_dict)
        for word, _ in counter:
            self.qa_dict[word] = len(self.qa_dict)

        self.qa_reverse_dict = {v: k for k, v in self.qa_dict.items()}

    # 语料向量化
    def text2id(self):
        self.qtext2id = []
        self.qtext2id_length = []
        self.atext2id_target = []
        self.atext2id_input = []
        self.atext2id_input_length = []

        for q in self.raw_q:
            tmp = []
            for word in q:
                tmp.append(self.qa_dict[word] if word in self.qa_dict.keys() else self.qa_dict['<UNK>'])
            self.qtext2id.append(tmp)
            self.qtext2id_length.append(len(tmp))

        for a in self.raw_a:
            tmp = []
            for word in a:
                tmp.append(self.qa_dict[word] if word in self.qa_dict.keys() else self.qa_dict['<UNK>'])

            tmp1 = tmp.copy()
            tmp1.insert(0, self.qa_dict['<GO>'])
            self.atext2id_input.append(tmp1)
            self.atext2id_input_length.append(len(tmp1))

            tmp.append(self.qa_dict['<EOS>'])
            self.atext2id_target.append(tmp)


class Encoder(torch.nn.Module):
    def __init__(self, l_dict_qa=1000):
        super(Encoder, self).__init__()
        self.l_dict_qa = l_dict_qa
        self.encoder_embedding = Embedding(l_dict_qa, params.embedding_qa_size)
        self.encoder_gru = GRU(input_size=params.embedding_qa_size, hidden_size=params.hidden_units,
                               num_layers=params.n_layers, dropout=params.drop_prob, batch_first=True)

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
        return torch.zeros(params.n_layers, batch_size, params.hidden_units)


class AttnDecoder(torch.nn.Module):
    def __init__(self, l_dict_qa):
        super(AttnDecoder, self).__init__()
        self.l_dict_qa = l_dict_qa

        self.decoder_embedding = Embedding(l_dict_qa, params.embedding_qa_size)
        self.decoder_gru = GRU(input_size=params.embedding_qa_size + params.hidden_units,
                               hidden_size=params.hidden_units,
                               num_layers=params.n_layers,
                               dropout=params.drop_prob,
                               batch_first=True)

        self.attn = Linear(params.n_layers * params.hidden_units, params.hidden_units, bias=False)
        self.project = Linear(params.hidden_units, l_dict_qa)

    def forward(self, input, hidden, encoder_hiddens):
        # input: B(=1)*L(=1)
        # hidden: n_layers*B(=1)*hidden_units
        # encoder_hiddens: B(=1)*encoder_L*hidden_units
        encoder_hiddens = torch.squeeze(encoder_hiddens, 0)
        ei = self.attn(hidden.transpose(1, 0).contiguous().view(1, -1))
        eij = F.softmax(ei.mm(encoder_hiddens.transpose(1, 0)), dim=1)
        ci = eij.mm(encoder_hiddens).view(1, 1, -1)
        inputs = torch.cat([ci, self.decoder_embedding(input)], dim=2)
        output, hidden = self.decoder_gru(inputs, hidden)
        output = self.project(output)
        return output, hidden


class Usr():
    def train(self, qtext2id, qtext2id_length, qa_dict, qa_reverse_dict,
              atext2id_input, atext2id_input_length, atext2id_target):
        encoder = Encoder(len(qa_dict))
        attndecoder = AttnDecoder(len(qa_dict))

        if params.mode == 'train1':
            encoder.load_state_dict(torch.load(params.model_save_path + 'chatbot1_encoder.pkl'))
            attndecoder.load_state_dict(torch.load(params.model_save_path + 'chatbot1_attndecoder.pkl'))

        loss = CrossEntropyLoss()
        paramters = list(encoder.parameters()) + list(attndecoder.parameters())
        optimizer = torch.optim.Adam(paramters, lr=params.lr)

        test = ['你真好', '吃完饭去干什么呢？']

        test_strip = [jieba.lcut(test[i]) for i in range(len(test))]
        test2id = []
        for i in range(len(test)):
            tmp = []
            for word in test_strip[i]:
                tmp.append(qa_dict[word] if word in qa_dict.keys() else qa_dict['<UNK>'])
            test2id.append(tmp)

        test2id_len = [len(test2id[i]) for i in range(len(test))]
        test_batch = test2id
        test_batch_len = test2id_len

        print(test_batch)
        print(test_batch_len)

        test_result_batch_len = [10, 12]

        idx = np.argsort(-np.array(test_batch_len))

        test = self.rearrange(test, idx)
        x_test_input_batch = self.rearrange(self.padding(test_batch, test_batch_len, qa_dict['<PAD>']), idx)
        x_test_input_batch_length = self.rearrange(test_batch_len, idx)
        y_test_input_batch_length = self.rearrange(test_result_batch_len, idx)

        m_samples = len(qtext2id)
        total_batch = m_samples // params.batch_size

        loss_ = []
        for epoch in range(1, params.epochs + 1):
            loss_epoch = 0.0
            for batch in range(total_batch):
                encoder.train()
                attndecoder.train()

                loss_batch = 0.0

                x_input_batch = qtext2id[batch * params.batch_size: (batch + 1) * params.batch_size]
                y_input_batch = atext2id_input[batch * params.batch_size: (batch + 1) * params.batch_size]
                y_target_batch = atext2id_target[batch * params.batch_size: (batch + 1) * params.batch_size]

                x_input_batch_length = qtext2id_length[batch * params.batch_size: (batch + 1) * params.batch_size]
                y_input_batch_length = atext2id_input_length[batch * params.batch_size: (batch + 1) * params.batch_size]

                idx = np.argsort(-np.array(x_input_batch_length))
                x_input_batch = self.rearrange(self.padding(x_input_batch, x_input_batch_length, qa_dict['<PAD>']), idx)
                y_input_batch = self.rearrange(self.padding(y_input_batch, y_input_batch_length, qa_dict['<PAD>']), idx)
                y_target_batch = self.rearrange(self.padding(y_target_batch, y_input_batch_length, qa_dict['<PAD>']),
                                                idx)

                x_input_batch_length = self.rearrange(x_input_batch_length, idx)
                y_input_batch_length = self.rearrange(y_input_batch_length, idx)

                encoder_initial_state = encoder.initial_hidden(params.batch_size)
                encoder_output, final_state = encoder(torch.from_numpy(np.array(x_input_batch, dtype=np.int64)),
                                                      x_input_batch_length, encoder_initial_state)
                for i in range(params.batch_size):
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
                loss_batch /= params.batch_size

                loss_epoch += loss_batch.data.numpy()

                encoder.zero_grad()
                attndecoder.zero_grad()
                loss_batch.backward()
                torch.nn.utils.clip_grad_value_(paramters, 5)
                optimizer.step()

                encoder.train(False)
                attndecoder.train(False)

                sys.stdout.write('>> %d/%d | %d/%d\n' % (epoch, params.epochs, batch + 1, total_batch))

                encoder_initial_state = encoder.initial_hidden(len(test))
                encoder_output, final_state = encoder(torch.from_numpy(np.array(x_test_input_batch, dtype=np.int64)),
                                                      x_test_input_batch_length,
                                                      encoder_initial_state)

                for i in range(len(test)):
                    sys.stdout.write('问: %s\n' % (test[i]))
                    result = []
                    decoder_input = torch.from_numpy(qa_dict['<GO>'] * np.ones([1, 1], dtype=np.int64))
                    decoder_state = final_state[:, i:i + 1]
                    H = encoder_output[i:i + 1, :x_test_input_batch_length[i]]
                    for j in range(y_test_input_batch_length[i]):
                        decoder_output, decoder_state = attndecoder(decoder_input,
                                                                    decoder_state,
                                                                    H)
                        decoder_output = torch.argmax(decoder_output, dim=2)
                        check_eos = decoder_output.numpy()[0, 0]
                        if check_eos == qa_dict['<EOS>']:
                            break
                        else:
                            result.append(qa_reverse_dict[check_eos])
                            decoder_input = decoder_output
                    sys.stdout.write('答: %s\n\n' % (''.join(result)))
                sys.stdout.flush()

            loss_epoch /= total_batch
            loss_.append(loss_epoch)

            print('\033[1;31;40m')
            print('>> %d/%d | Loss:%.9f' % (epoch, params.epochs, loss_[-1]))
            print('\033[0m')

            r = np.random.permutation(m_samples)
            qtext2id = self.rearrange(qtext2id, r)
            qtext2id_length = self.rearrange(qtext2id_length, r)
            atext2id_input = self.rearrange(atext2id_input, r)
            atext2id_input_length = self.rearrange(atext2id_input_length, r)
            atext2id_target = self.rearrange(atext2id_target, r)

            if epoch % params.per_save == 0:
                torch.save(encoder.state_dict(), params.model_save_path + 'chatbot1_encoder.pkl')
                torch.save(attndecoder.state_dict(), params.model_save_path + 'chatbot1_attndecoder.pkl')

        fig = plt.figure(figsize=(10, 8))
        plt.plot(loss_)
        plt.savefig(params.model_save_path + 'chatbot1_loss.png')
        plt.close(fig)

    def predict(self, qa_dict, qa_reverse_dict):
        encoder = Encoder(len(qa_dict))
        attndecoder = AttnDecoder(len(qa_dict))

        encoder.load_state_dict(torch.load(params.model_save_path + 'chatbot1_encoder.pkl'))
        attndecoder.load_state_dict(torch.load(params.model_save_path + 'chatbot1_attndecoder.pkl'))
        encoder.train(False)
        attndecoder.train(False)

        while 1:
            print('\033[1;31;40m')

            test = input('问: ')
            test_strip = [jieba.lcut(test)]
            test_len = [len(test_strip[0])]
            test2id = []
            tmp = []
            for word in test_strip[0]:
                tmp.append(qa_dict[word] if word in qa_dict.keys() else qa_dict['<UNK>'])
            test2id.append(tmp)

            x_test_input_batch = test2id
            x_test_input_batch_length = test_len
            y_test_input_batch_length = [20]

            encoder_initial_state = encoder.initial_hidden(1)
            encoder_output, final_state = encoder(torch.from_numpy(np.array(x_test_input_batch, dtype=np.int64)),
                                                  x_test_input_batch_length,
                                                  encoder_initial_state)

            result = []
            decoder_input = torch.from_numpy(qa_dict['<GO>'] * np.ones([1, 1], dtype=np.int64))
            decoder_state = final_state[:, i:i + 1]
            H = encoder_output[0:1, :x_test_input_batch_length[0]]
            for j in range(y_test_input_batch_length[0]):
                decoder_output, decoder_state = attndecoder(decoder_input,
                                                            decoder_state,
                                                            H)
                decoder_output = torch.argmax(decoder_output, dim=2)
                check_eos = decoder_output.numpy()[0, 0]
                if check_eos == qa_dict['<EOS>']:
                    break
                else:
                    result.append(qa_reverse_dict[check_eos])
                    decoder_input = decoder_output
            sys.stdout.write('答: %s\n\n' % (''.join(result)))
            sys.stdout.flush()

    def padding(self, x, l, padding_id):
        l_max = np.max(l)
        return [x[i] + [padding_id] * (l_max - l[i]) for i in range(len(x))]

    def rearrange(self, x, r):
        return [x[ri] for ri in r]


def main():
    if not os.path.exists(params.model_save_path):
        os.makedirs(params.model_save_path)

    lang = Lang()
    print(len(lang.raw_q), lang.raw_q[:10])
    print(len(lang.raw_a), lang.raw_a[:10])

    print(len(lang.qa_dict), [lang.qa_reverse_dict[i] for i in range(100)], '\n')

    print(lang.qtext2id_length[:10], np.max(lang.qtext2id_length), np.min(lang.qtext2id_length),
          np.argmin(lang.qtext2id_length), lang.qtext2id[:10], '\n')
    print(lang.atext2id_input_length[:10], np.max(lang.atext2id_input_length), np.min(lang.atext2id_input_length),
          np.argmin(lang.atext2id_input_length), lang.atext2id_target[:10], '\n')
    print(lang.atext2id_input_length[:10], np.max(lang.atext2id_input_length), np.min(lang.atext2id_input_length),
          np.argmin(lang.atext2id_input_length), lang.atext2id_input[:10], '\n')

    if params.mode.startswith('train'):
        f = open(sys.path[0] + '/xiaohuangji.txt', 'w+')
        for i in range(len(lang.qtext2id)):
            f.write(' '.join(lang.raw_q[i]) + '\n')
            f.write(' '.join(lang.raw_a[i]) + '\n')
            f.write('\n')
        f.close()

    usr = Usr()
    if params.mode == 'predict':
        usr.predict(lang.qa_dict, lang.qa_reverse_dict)
    else:
        usr.train(lang.qtext2id, lang.qtext2id_length, lang.qa_dict, lang.qa_reverse_dict, lang.atext2id_input,
                  lang.atext2id_input_length, lang.atext2id_target)


if __name__ == '__main__':
    main()
