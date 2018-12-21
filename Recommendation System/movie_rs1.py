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
        users = pd.read_table(self.config.datadir + '/users.dat', sep='::', header=None, names=usernames,
                              engine='python')
        users = users.drop(['zip'], axis=1)

        movienames = ['movie_id', 'title', 'genres']
        movies = pd.read_table(self.config.datadir + '/movies.dat', sep='::', header=None, names=movienames,
                               engine='python')

        ratingnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table(self.config.datadir + '/ratings.dat', sep='::', header=None, names=ratingnames,
                                engine='python')
        ratings = ratings.drop(['timestamp'], axis=1)

        len_users = len(users)
        self.user_id_dict = {users['user_id'][i].astype(np.int32): i for i in range(len_users)}
        print(self.user_id_dict)

        self.users = {'user_id': np.array(users['user_id'], dtype=np.int32),
                      'gender': np.array([[1, 0] if users['gender'][i] == 'F' else [0, 1] for i in range(len_users)],
                                         dtype=np.int32),
                      'age': np.array([self.age_dict[users['age'][i]] for i in range(len_users)]),
                      'occupation': np.array([self.occupation_dict[users['occupation'][i]] for i in range(len_users)])}

        len_movies = len(movies)
        self.movie_id_dict = {movies['movie_id'][i].astype(np.int32): i for i in range(len_movies)}
        print(self.movie_id_dict)

        title_len, title = self.get_title([movies['title'][i] for i in range(len_movies)])
        self.movies = {'movie_id': np.array(movies['movie_id'], dtype=np.int32),
                       'title': title,
                       'title_len': title_len,
                       'genres': np.array([np.sum([self.genres_dict[movies['genres'][i].split('|')[j]] for j in
                                                   range(len(movies['genres'][i].split('|')))], axis=0) for i in
                                           range(len_movies)])}

        self.ratings = {'user_id': np.array(ratings['user_id'], dtype=np.int32),
                        'movie_id': np.array(ratings['movie_id'], dtype=np.int32),
                        'rating': np.array(ratings['rating'], dtype=np.float32)}

        print(self.users)
        print(self.movies)
        print(self.ratings)

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
        self.build()

    def build(self):
        self.user_gender = tf.placeholder(tf.float32, [None, 2], name='gender')
        self.user_age = tf.placeholder(tf.float32, [None, 7], name='age')
        self.user_occupation = tf.placeholder(tf.float32, [None, 21], name='occupation')
        self.movie_title = tf.placeholder(tf.int32, [None, None], name='title')
        self.movie_title_len = tf.placeholder(tf.int32, [None], name='len')
        self.movie_genres = tf.placeholder(tf.float32, [None, 18], name='genres')
        self.rating = tf.placeholder(tf.float32, [None], name='rating')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        user_feature1 = tf.layers.dense(tf.concat([self.user_gender, self.user_age, self.user_occupation], axis=1),
                                        self.config.user_unit1)
        user_feature = tf.layers.dense(user_feature1, self.config.user_and_movie_unit)

        movie_feature1 = tf.layers.dense(self.movie_genres, self.config.movie_unit1)
        embed_matrix = tf.Variable(tf.random_uniform([self.dict_size, self.config.embed_size], -1.0, 1.0))
        embed_title = tf.nn.embedding_lookup(embed_matrix, self.movie_title, name='embed_title')
        _, movie_feature2 = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(self.config.gru_unit),
                                               output_keep_prob=self.keep_prob),
            inputs=embed_title,
            sequence_length=self.movie_title_len,
            dtype=tf.float32)
        movie_feature = tf.layers.dense(tf.concat([movie_feature1, movie_feature2], axis=1),
                                        self.config.user_and_movie_unit)

        self.inference = tf.reduce_sum(tf.multiply(user_feature, movie_feature), axis=1)
        self.loss = tf.nn.l2_loss(self.inference - self.rating, name='loss')

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph(), filename_suffix='recommsys')
        writer.flush()
        writer.close()

    def train(self, lang):
        m_sample = np.shape(lang.ratings['rating'])[0]

        total_batch = m_sample // self.config.batch_size

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
                user_batch = np.array([lang.user_id_dict[i] for i in user_id_batch])

                movie_id_batch = lang.ratings['movie_id'][
                                 batch * self.config.batch_size:(batch + 1) * self.config.batch_size]
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


def main(unused_argv):
    lang = Lang(CONFIG)
    rs = RecommendationSystem(CONFIG, lang.dict_len)
    rs.train(lang)


if __name__ == '__main__':
    tf.app.run()

