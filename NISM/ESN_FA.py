import numpy as np
import pandas as pd
import scipy.io
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore
from scipy import sparse
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from random import random,uniform,choice,randint
from time import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import sys
from keras.utils.np_utils import to_categorical

class Reservoir():
    def __init__(self, n_internal_units = 100, spectral_radius = 0.99, leak = None,
                 connectivity = 0.3, input_scaling = 0.2, noise_level = 0.01, circle = False):
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak

        self._input_weights = None
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                n_internal_units,
                spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units,
                connectivity,
                spectral_radius
            )

    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):
        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0,-1] = spectral_radius
        for i in range(n_internal_units-1):
            internal_weights[i+1, i] = spectral_radius

        return internal_weights

    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius):
        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        internal_weights[np.where(internal_weights>0)] -= 0.5

        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /=np.abs(e_max)/spectral_radius

        return internal_weights

    def _compute_state_matrix(self, X, n_drop=0):
        N, T, _ = X.shape
        previous_state = np.zeros((N, self._n_internal_units), dtype=float)

        # Storage
        state_matrix = np.empty((N, T - n_drop, self._n_internal_units), dtype=float)
        for t in range(T):
            current_input = X[:, t, :]

            # Calculate state
            state_before_tanh = self._internal_weights.dot(previous_state.T) + self._input_weights.dot(current_input.T)

            # Add noise
            state_before_tanh += np.random.rand(self._n_internal_units, N) * self._noise_level

            # Apply nonlinearity and leakage (optional)
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T
            else:
                previous_state = (1.0 - self._leak) * previous_state + np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if (t > n_drop - 1):
                state_matrix[:, t - n_drop, :] = previous_state

        return state_matrix

    def get_states(self, X, n_drop=0, bidir=True):
        N, T, V = X.shape
        if self._input_weights is None:
            self._input_weights = (2.0 * np.random.binomial(1, 0.5,
                                                            [self._n_internal_units, V]) - 1.0) * self._input_scaling

        # compute sequence of reservoir states
        states = self._compute_state_matrix(X, n_drop)

        # reservoir states on time reversed input
        if bidir is True:
            X_r = X[:, ::-1, :]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states

    def getReservoirEmbedding(self, X, pca, ridge_embedding, n_drop=0, bidir=True, test=False):

        res_states = self.get_states(X, n_drop=0, bidir=True)

        N_samples = res_states.shape[0]
        res_states = res_states.reshape(-1, res_states.shape[2])
        # ..transform..
        if test:
            red_states = pca.transform(res_states)
        else:
            red_states = pca.fit_transform(res_states)
            # ..and put back in tensor form
        red_states = red_states.reshape(N_samples, -1, red_states.shape[1])

        coeff_tr = []
        biases_tr = []

        for i in range(X.shape[0]):
            ridge_embedding.fit(red_states[i, 0:-1, :], red_states[i, 1:, :])
            coeff_tr.append(ridge_embedding.coef_.ravel())
            biases_tr.append(ridge_embedding.intercept_.ravel())
        # print(np.array(coeff_tr).shape,np.array(biases_tr).shape)
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        return input_repr

def func(n, m, e, f):
    df = pd.read_csv('NISM.csv')
    df1 = df[df['label'] == 'DNS'].head(n=int(df[df['label'] == 'DNS'].shape[0] * 0.1))
    print(df1)
    print('\n')
    df1 = df1.append(df[df['label'] == 'lime'].head(n=int(df[df['label'] == 'lime'].shape[0] * 0.006)))
    df1 = df1.append(df[df['label'] == 'FTP'])
    df1 = df1.append(df[df['label'] == 'HTTP'].head(n=int(df[df['label'] == 'HTTP'].shape[0] * 0.2)))
    df1 = df1.append(df[df['label'] == 'TELNET'])
    df1 = df1.append(df[df['label'] == 'localForwarding'])
    df1 = df1.append(df[df['label'] == 'remoteForwarding'])
    df1 = df1.append(df[df['label'] == 'scp'])
    df1 = df1.append(df[df['label'] == 'sftp'])
    df1 = df1.append(df[df['label'] == 'shell'])
    df = df1.append(df[df['label'] == 'x11'])

    print(df.head())
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(cols)
    sc = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    sc_df = pd.DataFrame(sc, columns=cols)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    cat = encoder.fit_transform(df['label'])
    from sklearn.model_selection import train_test_split
    sc_traindf, sc_testdf, traincat, testcat = train_test_split(sc_df, cat,
                                                                test_size=0.5,
                                                                random_state=0)
    traincat = to_categorical(traincat)
    testcat = to_categorical(testcat)

    # 绘制流量分布图
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=0)
    traffic_class_train = df_train[['label']].apply(lambda x: x.value_counts())
    print(traffic_class_train)
    traffic_class_test = df_test[['label']].apply(lambda x: x.value_counts())
    print(traffic_class_test)
    traffic_class_dist = pd.concat([traffic_class_train, traffic_class_test], axis=1)
    print(traffic_class_dist)
    plt.figure(figsize=(10, 5))
    plot = traffic_class_dist.plot(kind='bar')
    plot.set_title("Traffic Class", fontsize=20)
    plot.grid(color='lightgray', alpha=0.5)
    plt.legend(['train data', 'test data'])
    plt.show()
    enctrain = pd.DataFrame(traincat,
                            columns=['250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260'])
    refclasscol = pd.concat([sc_traindf, enctrain], axis=1).columns
    refclass = np.concatenate((sc_traindf.values, enctrain.values), axis=1)
    X = refclass

    X = sc_traindf
    y = traincat
    X = np.reshape(X.values, (X.values.shape[0], X.values.shape[1], 1))
    sc_testdf = np.reshape(sc_testdf.values, (sc_testdf.values.shape[0], sc_testdf.values.shape[1], 1))
    res = Reservoir(n_internal_units=n, spectral_radius=m, leak=0.6,
                    connectivity=e, input_scaling=f, noise_level=0.01, circle=False)
    pca = PCA()
    ridge_embedding = Ridge(alpha=10, fit_intercept=True)
    readout = Ridge(alpha=5)
    input_repr = res.getReservoirEmbedding(np.array(X), pca, ridge_embedding, n_drop=5, bidir=True, test=False)
    print(input_repr)

    input_repr_te = res.getReservoirEmbedding(np.array(sc_testdf), pca, ridge_embedding, n_drop=5, bidir=True,
                                              test=False)
    print(input_repr_te)

    readout.fit(input_repr, y)
    logits = readout.predict(input_repr_te)
    predict_target2 = np.argmax(logits, axis=1)
    print(predict_target2)

    testcat = np.argmax(testcat, axis=1)
    a = accuracy_score(testcat, predict_target2)
    print(a)
    from sklearn import metrics

    print("预测正确数量，测试集样本量：")
    print(sum(predict_target2 == testcat), len(testcat))
    print("精确度等指标：")
    print(metrics.classification_report(testcat, predict_target2))
    print("混淆矩阵：")
    print(metrics.confusion_matrix(testcat, predict_target2))

    confusion = metrics.confusion_matrix(testcat, predict_target2)
    return a



class ffa():
    def __init__(self, rnges, population, alpha=0.2, gamma=1.0):
        self.population = population
        self.alpha = 0.5
        self.gamma = 1.0
        self.rnges = rnges
        self.xn = np.zeros(0)
        self.yn = np.zeros(0)
        self.zn = np.zeros(0)
        self.wn = np.zeros(0)
        self.fireflies = list()
        self.lightn = np.zeros(0)

    def collect_fireflies(self):
        print("Collecting fireflies")
        self.fireflies = list()
        for i in range(len(self.xn)):
            self.fireflies.append(func(self.xn[i], self.yn[i], self.zn[i], self.wn[i]))
            print(self.fireflies)

    def initiate(self, max_gen):
        print("Called ffa initiate")
        spectral_radius_range = self.rnges['spectral_radius_upper'] - self.rnges['spectral_radius_lower']
        connectivity_range = self.rnges['connectivity_upper'] - self.rnges['connectivity_lower']
        input_scaling_range = self.rnges['input_scaling_upper'] - self.rnges['input_scaling_lower']
        self.xn = np.random.randint(low=self.rnges['n_internal_units_lower'], high=self.rnges['n_internal_units_upper'], size=self.population)
        self.yn = np.random.rand(self.population)*spectral_radius_range+self.rnges['spectral_radius_lower']
        self.zn = np.random.rand(self.population)*connectivity_range+self.rnges['connectivity_lower']
        self.wn = np.random.rand(self.population)*input_scaling_range+self.rnges['input_scaling_lower']
        self.collect_fireflies()
        self.lightn = np.zeros(self.yn.shape)

    def findrange(self):
        for i in range(self.yn.size):
            if self.xn[i] <= self.rnges['n_internal_units_lower']:
                self.xn[i] = self.rnges['n_internal_units_lower']
            if self.xn[i] >= self.rnges['n_internal_units_upper']:
                self.xn[i] = self.rnges['n_internal_units_upper']
            if self.yn[i] <= self.rnges['spectral_radius_lower']:
                self.yn[i] = self.rnges['spectral_radius_lower']
            if self.yn[i] >= self.rnges['spectral_radius_upper']:
                self.yn[i] = self.rnges['spectral_radius_upper']
            if self.zn[i] <= self.rnges['connectivity_lower']:
                self.zn[i] = self.rnges['connectivity_lower']
            if self.zn[i] >= self.rnges['connectivity_upper']:
                self.zn[i] = self.rnges['connectivity_upper']
            if self.wn[i] <= self.rnges['input_scaling_lower']:
                self.wn[i] = self.rnges['input_scaling_lower']
            if self.wn[i] >= self.rnges['input_scaling_upper']:
                self.wn[i] = self.rnges['input_scaling_upper']

    def ffa_move(self, xo, yo, zo, wo, lighto):
        ni = self.yn.shape[0]
        nj = yo.shape[0]
        for i in range(ni):
            for j in range(nj):
                r1 = np.sqrt((self.xn[i] - xo[j]) ** 2 + (self.yn[i] - yo[j]) ** 2 +
                             (self.zn[i] - zo[j]) ** 2 + (self.wn[i] - wo[j]) ** 2)
                if self.lightn[i] < lighto[j]:
                    beta0 = 1
                    beta1 = beta0 * np.exp(-1 * self.gamma * r1 ** 2)
                    self.xn[i] = self.xn[i] * (1 - beta1) + xo[j] * beta1 + self.alpha * (
                            np.random.randint(low=self.rnges['n_internal_units_lower'],
                                              high=self.rnges['n_internal_units_upper']) - 10)
                    self.yn[i] = self.yn[i] * (1 - beta1) + yo[j] * beta1 + self.alpha * (np.random.rand() - 0.5)
                    self.zn[i] = self.zn[i] * (1 - beta1) + zo[j] * beta1 + self.alpha * (np.random.rand() - 0.5)
                    self.wn[i] = self.wn[i] * (1 - beta1) + wo[j] * beta1 + self.alpha * (np.random.rand() - 0.5)

        print("INSIDE MOVE!")
        print(self.xn, self.yn, self.zn, self.wn)
        self.findrange()
        self.collect_fireflies()
        print("again insode MOVWE")
        print(self.xn, self.yn, self.zn, self.wn)
        print(self.fireflies)

    def firefly_simple(self, max_gen):
        self.initiate(max_gen)
        print("Inside ffa_simple")
        for i in range(max_gen):
            print(self.xn)
            print(self.yn)
            print(self.zn)
            print(self.wn)
            qn = []
            for j in range(len(self.xn)):
                qn.append(self.fireflies[j])
            lighto = np.sort(qn)
            indexes = np.argsort(qn)
            # lighto = lighto[::-1]         # for minima
            # indexes = indexes[::-1]       # for minima
            print("\n At step " + str(i) + "with values- ", qn)
            xo = np.array([self.xn[j] for j in indexes])
            yo = np.array([self.yn[j] for j in indexes])
            zo = np.array([self.zn[j] for j in indexes])
            wo = np.array([self.wn[j] for j in indexes])
            print(self.xn, xo)
            print(self.yn, yo)
            print(self.zn, zo)
            print(self.wn, wo)
            # print(type(zo),type(self.zn))
            self.ffa_move(xo, yo, zo, wo, lighto)
            print("\n\n Moved " + str(i))



if __name__ == '__main__':
    rnges = {'n_internal_units_lower':20, 'n_internal_units_upper':50, 'spectral_radius_lower':0, 'spectral_radius_upper':1,
             'connectivity_lower':0, 'connectivity_upper':1, 'input_scaling_lower':0, 'input_scaling_upper':1}

    obj = ffa(rnges=rnges, population=5)
    obj.firefly_simple(max_gen=20)