'''
    Movie Recommendation System based on Deep Learning
    Data : MovieLens 1M
'''
import tensorflow as tf
import pandas as pd
from rs_hp import CONFIG
import numpy as np
import collections
import sys


class Lang():
    def __init__(self, config):
        self.config = config
        self.age_dict = {1: self.onehot(0, 7), 18: self.onehot(1, 7), 25: self.onehot(2, 7), 35: self.onehot(3, 7),
                         45: self.onehot(4, 7), 50: self.onehot(5, 7), 56: self.onehot(6, 7)}
        self.occupation_dict = {i: self.onehot(i, 21) for i in range(21)}

        self.genres_dict = {'Action': self.onehot(0, 18), 'Adventure': self.onehot(1, 18),
                            'Animation': self.onehot(2, 18), 'Children\'s': self.onehot(3, 18),
                            'Comedy': self.onehot(4, 18), 'Crime': self.onehot(5, 18),
                            'Documentary': self.onehot(6, 18), 'Drama': self.onehot(7, 18),
                            'Fantasy': self.onehot(8, 18), 'Film-Noir': self.onehot(9, 18),
                            'Horror': self.onehot(10, 18), 'Musical': self.onehot(11, 18),
                            'Mystery': self.onehot(12, 18), 'Romance': self.onehot(13, 18),
                            'Sci-Fi': self.onehot(14, 18), 'Thriller': self.onehot(15, 18),
                            'War': self.onehot(16, 18), 'Western': self.onehot(17, 18)}
        self.readdata()

    def readdata(self):
        usernames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        self.users_table = pd.read_table(self.config.datadir + '/users.dat', sep='::', header=None, names=usernames,
                                         engine='python')

        movienames = ['movie_id', 'title', 'genres']
        self.movies_table = pd.read_table(self.config.datadir + '/movies.dat', sep='::', header=None, names=movienames,
                                          engine='python')

        ratingnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table(self.config.datadir + '/ratings.dat', sep='::', header=None, names=ratingnames,
                                engine='python')

        len_users = len(self.users_table)
        self.user_id_dict = {self.users_table['user_id'][i].astype(np.int32): i for i in range(len_users)}

        self.users = {'user_id': np.array(self.users_table['user_id'], dtype=np.int32),
                      'gender': np.array(
                          [[1, 0] if self.users_table['gender'][i] == 'F' else [0, 1] for i in range(len_users)],
                          dtype=np.int32),
                      'age': np.array([self.age_dict[self.users_table['age'][i]] for i in range(len_users)]),
                      'occupation': np.array(
                          [self.occupation_dict[self.users_table['occupation'][i]] for i in range(len_users)])}

        len_movies = len(self.movies_table)
        self.movie_id_dict = {self.movies_table['movie_id'][i].astype(np.int32): i for i in range(len_movies)}

        title_len, title = self.get_title([self.movies_table['title'][i] for i in range(len_movies)])
        self.movies = {'movie_id': np.array(self.movies_table['movie_id'], dtype=np.int32),
                       'title': title,
                       'title_len': title_len,
                       'genres': np.array(
                           [np.sum([self.genres_dict[self.movies_table['genres'][i].split('|')[j]] for j in
                                    range(len(self.movies_table['genres'][i].split('|')))], axis=0) for i in
                            range(len_movies)])}

        self.ratings = {'user_id': np.array(ratings['user_id'], dtype=np.int32),
                        'movie_id': np.array(ratings['movie_id'], dtype=np.int32),
                        'rating': np.array(ratings['rating'], dtype=np.float32)}

    def get_title(self, title_list):
        lt = len(title_list)

        title_list = self.split_title(title_list)

        tmp = []

        for i in range(lt):
            tmp.extend(title_list[i])

        self.create_dict(tmp)

        title_len = np.array([len(title_list[i]) for i in range(lt)])

        len_max = np.max(title_len)

        tmp1 = []
        for title in title_list:
            tmp2 = []
            for word in title:
                tmp2.append(self.title_dict[word])
            if len(title) < len_max:
                tmp2.extend([0] * (len_max - len(title)))
            tmp1.append(tmp2)
        return title_len, np.array(tmp1)

    def create_dict(self, tmp):
        counter = collections.Counter(tmp).most_common()

        self.title_dict = dict()

        self.title_dict['<pad>'] = len(self.title_dict)
        for word, _ in counter:
            self.title_dict[word] = len(self.title_dict)

        self.dict_len = len(self.title_dict)

    def split_title(self, title):
        lt = len(title)
        tmp = []
        for i in range(lt):
            tmp.append(title[i].lower().split(',')[0].split('(')[0].strip().split(' '))
        return tmp

    def onehot(self, i, l):
        tmp = np.zeros(l, dtype=np.int32)
        tmp[i] = 1
        return tmp


class RecommendationSystem():
    def __init__(self, config, dict_size):
        self.config = config
        self.dict_size = dict_size

    def build(self):
        with tf.name_scope('Input'):
            self.user_gender = tf.placeholder(tf.float32, [None, 2], name='gender')
            self.user_age = tf.placeholder(tf.float32, [None, 7], name='age')
            self.user_occupation = tf.placeholder(tf.float32, [None, 21], name='occupation')
            self.movie_title = tf.placeholder(tf.int32, [None, None], name='title')
            self.movie_title_len = tf.placeholder(tf.int32, [None], name='len')
            self.movie_genres = tf.placeholder(tf.float32, [None, 18], name='genres')
            self.rating = tf.placeholder(tf.float32, [None], name='rating')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.lr = tf.placeholder(tf.float32, name='learning_rate')

        with tf.name_scope('user'):
            user_feature1 = tf.layers.dense(tf.concat([self.user_gender, self.user_age, self.user_occupation], axis=1),
                                            self.config.user_unit1, activation=tf.nn.relu)
            user_feature = tf.layers.dense(user_feature1, self.config.user_and_movie_unit, activation=tf.nn.relu,
                                           name='user_feature')

        with tf.name_scope('movie'):
            movie_feature1 = tf.layers.dense(self.movie_genres, self.config.movie_unit1, activation=tf.nn.relu)
            embed_matrix = tf.Variable(tf.random_uniform([self.dict_size, self.config.embed_size], -1.0, 1.0))
            embed_title = tf.nn.embedding_lookup(embed_matrix, self.movie_title, name='embed_title')
            _, movie_feature2 = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.config.gru_unit),
                                                   output_keep_prob=self.keep_prob),
                inputs=embed_title,
                sequence_length=self.movie_title_len,
                dtype=tf.float32)
            movie_feature = tf.layers.dense(tf.concat([movie_feature1, movie_feature2], axis=1),
                                            self.config.user_and_movie_unit, activation=tf.nn.relu,
                                            name='movie_feature')

        with tf.name_scope('Inference'):
            self.score = tf.add(tf.multiply(4.0, tf.sigmoid(
                tf.reduce_sum(tf.multiply(user_feature, movie_feature), axis=1))), 1.0, name='score')

        with tf.name_scope('Loss'):
            self.loss = tf.sqrt(tf.losses.mean_squared_error(self.score, self.rating), name='loss')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        if self.config.graph_write:
            writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='recommsys')
            writer.flush()
            writer.close()

    def train(self, lang):
        m_samples = np.shape(lang.ratings['rating'])[0]

        total_batch = m_samples // self.config.batch_size

        loss_ = []

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=1)

        for epoch in range(1, self.config.epochs + 1):
            loss_epoch = 0.0
            for batch in range(total_batch):
                sys.stdout.write('\r>> %d/%d | %d/%d' % (epoch, self.config.epochs, batch + 1, total_batch))
                sys.stdout.flush()

                user_id_batch = lang.ratings['user_id'][
                                batch * self.config.batch_size:(batch + 1) * self.config.batch_size]
                movie_id_batch = lang.ratings['movie_id'][
                                 batch * self.config.batch_size:(batch + 1) * self.config.batch_size]

                user_batch = np.array([lang.user_id_dict[i] for i in user_id_batch])
                movie_batch = np.array([lang.movie_id_dict[i] for i in movie_id_batch])

                gender_batch = lang.users['gender'][user_batch]
                age_batch = lang.users['age'][user_batch]
                occupation_batch = lang.users['occupation'][user_batch]

                title_batch = lang.movies['title'][movie_batch]
                title_len_batch = lang.movies['title_len'][movie_batch]
                genres_batch = lang.movies['genres'][movie_batch]

                rating_batch = lang.ratings['rating'][
                               batch * self.config.batch_size:(batch + 1) * self.config.batch_size]

                feed_dict = {self.user_gender: gender_batch, self.user_age: age_batch,
                             self.user_occupation: occupation_batch,
                             self.movie_title: title_batch, self.movie_title_len: title_len_batch,
                             self.movie_genres: genres_batch,
                             self.rating: rating_batch,
                             self.keep_prob: self.config.keep_prob,
                             self.lr: self.config.lr}
                _, loss_batch = sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                loss_epoch += loss_batch
            loss_.append(loss_epoch / total_batch)

            sys.stdout.write(' | Loss:%.9f\n' % (loss_[-1]))
            sys.stdout.flush()

            r = np.random.permutation(m_samples)
            lang.ratings['user_id'] = lang.ratings['user_id'][r]
            lang.ratings['movie_id'] = lang.ratings['movie_id'][r]
            lang.ratings['rating'] = lang.ratings['rating'][r]

            if epoch % self.config.per_save == 0:
                saver.save(sess, self.config.model_save_path + 'rs1')
                print('Model saved successfully!')

    def infer(self, lang, user_id):
        sess = tf.Session()
        new_saver = tf.train.import_meta_graph(self.config.model_save_path + 'rs1.meta')
        new_saver.restore(sess, self.config.model_save_path + 'rs1')

        graph = tf.get_default_graph()

        user_gender = graph.get_operation_by_name('Input/gender').outputs[0]
        user_age = graph.get_operation_by_name('Input/age').outputs[0]
        user_occupation = graph.get_operation_by_name('Input/occupation').outputs[0]
        movie_title = graph.get_operation_by_name('Input/title').outputs[0]
        movie_title_len = graph.get_operation_by_name('Input/len').outputs[0]
        movie_genres = graph.get_operation_by_name('Input/genres').outputs[0]
        keep_prob = graph.get_operation_by_name('Input/keep_prob').outputs[0]

        score = graph.get_tensor_by_name('Inference/score:0')

        len_movie = len(lang.movie_id_dict)

        user_id_batch = user_id * np.ones(len_movie, dtype=np.int32)
        movie_id_batch = lang.movies['movie_id']

        user_batch = np.array([lang.user_id_dict[i] for i in user_id_batch])
        movie_batch = np.array([lang.movie_id_dict[i] for i in movie_id_batch])

        gender_batch = lang.users['gender'][user_batch]
        age_batch = lang.users['age'][user_batch]
        occupation_batch = lang.users['occupation'][user_batch]

        title_batch = lang.movies['title'][movie_batch]
        title_len_batch = lang.movies['title_len'][movie_batch]
        genres_batch = lang.movies['genres'][movie_batch]

        feed_dict = {user_gender: gender_batch, user_age: age_batch,
                     user_occupation: occupation_batch,
                     movie_title: title_batch, movie_title_len: title_len_batch,
                     movie_genres: genres_batch,
                     keep_prob: 1.0}
        score_ = sess.run(score, feed_dict=feed_dict)

        score_top_k = -np.sort(-score_)[:self.config.top_k]
        index_top_k = np.argsort(-score_)
        for i in range(self.config.top_k):
            print('%5d %50s %40s     %.1f' % (
                lang.movies_table['movie_id'][index_top_k[i]],
                lang.movies_table['title'][index_top_k[i]],
                lang.movies_table['genres'][index_top_k[i]],
                score_top_k[i]))


def main(unused_argv):
    lang = Lang(CONFIG)
    rs = RecommendationSystem(CONFIG, lang.dict_len)
    if CONFIG.mode == 'train':
        rs.build()
        rs.train(lang)
    elif CONFIG.mode == 'infer':
        rs.infer(lang, 1)


if __name__ == '__main__':
    tf.app.run()
