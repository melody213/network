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
from pso import Pso

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
        internal_weights = np.zeros((n_internal_units,n_internal_units))
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
        N, T = X.shape
        previous_state = np.zeros((N, self._n_internal_units),dtype=float)
        state_matrix = np.empty((N, T-n_drop, self._n_internal_units), dtype=float)
        for t in range(T):
            X_row = X.shape[0]
            current_input = X[:,t].reshape(X_row,1)
            new_input_weights = self._input_weights.reshape(self._n_internal_units, 1)
            state_before_tanh = self._internal_weights.dot(previous_state.T) + new_input_weights.dot(current_input.T)

            state_before_tanh += np.random.rand(self._n_internal_units, N) * self._noise_level
            if self._leak is None:
                previous_state = np.tanh(state_before_tanh).T

            else:
                previous_state = (1.0 - self._leak)*previous_state + np.tanh(state_before_tanh).T

            if (t > n_drop - 1):
                state_matrix[:, t - n_drop] = previous_state

        return state_matrix

    def get_states(self, X, n_drop=0, bidir=True):
        N, T = X.shape
        if self._input_weights is None:
            self._input_weights = (2.0*np.random.binomial(1, 0.5, [self._n_internal_units]) - 1.0)*self._input_scaling

        states = self._compute_state_matrix(X, n_drop)

        if bidir is True:
            X_r = X[:, ::-1]
            states_r = self._compute_state_matrix(X_r, n_drop)
            states = np.concatenate((states, states_r), axis=2)

        return states

    def getReservoirEmbedding(self, X, pca, ridge_embedding, n_drop=5, bidir=True, test = False):
        res_states = self.get_states(X, n_drop=5, bidir=True)

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
            ridge_embedding.fit(red_states[i, 0:-1], red_states[i, 1:])
            coeff_tr.append(ridge_embedding.coef_.ravel())
            biases_tr.append(ridge_embedding.intercept_.ravel())
        # print(np.array(coeff_tr).shape,np.array(biases_tr).shape)
        input_repr = np.concatenate((np.vstack(coeff_tr), np.vstack(biases_tr)), axis=1)
        return input_repr



def func(x):
    df = pd.read_csv('data5.txt')
    df1 = df[df['250'] == 'WWW'].head(n=int(df[df['250'] == 'WWW'].shape[0] * 0.01))
    print(df1)
    print('\n')
    df1 = df1.append(df[df['250'] == 'MAIL'].head(n=int(df[df['250'] == 'MAIL'].shape[0] * 0.1)))
    df1 = df1.append(df[df['250'] == 'FTP-CONTROL'].head(n=int(df[df['250'] == 'FTP-CONTROL'].shape[0] * 0.1)))
    df1 = df1.append(df[df['250'] == 'FTP-DATA'].head(n=int(df[df['250'] == 'FTP-DATA'].shape[0] * 0.5)))
    df1 = df1.append(df[df['250'] == 'ATTACK'])
    df1 = df1.append(df[df['250'] == 'SERVICES'])
    df1 = df1.append(df[df['250'] == 'P2P'])
    df1 = df1.append(df[df['250'] == 'DATABASE'])
    df1 = df1.append(df[df['250'] == 'MULTIMEDIA'])
    df = df1.append(df[df['250'] == 'FTP-PASV'])

    print(df.head())
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(cols)
    sc = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    sc_df = pd.DataFrame(sc, columns=cols)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    cat = encoder.fit_transform(df['250'])
    from sklearn.model_selection import train_test_split
    sc_traindf, sc_testdf, traincat, testcat = train_test_split(sc_df, cat,
                                                                test_size=0.5,
                                                                random_state=0)

    # 绘制流量分布图
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=0)
    traffic_class_train = df_train[['250']].apply(lambda x: x.value_counts())
    print(traffic_class_train)
    traffic_class_test = df_test[['250']].apply(lambda x: x.value_counts())
    print(traffic_class_test)
    traffic_class_dist = pd.concat([traffic_class_train, traffic_class_test], axis=1)
    print(traffic_class_dist)
    plt.figure(figsize=(10, 5))
    plot = traffic_class_dist.plot(kind='bar')
    plot.set_title("Traffic Class", fontsize=20)
    plot.grid(color='lightgray', alpha=0.5)
    plt.legend(['train data', 'test data'])
    plt.show()

    X = sc_traindf
    y = traincat
    n, m, e, f = x[0], x[1], x[2], x[3]
    res = Reservoir(n_internal_units=int(n), spectral_radius=m, leak=0.6,
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
    pred_class = readout.predict(input_repr_te)
    print(pred_class)
    predictions = [int(round(x)) for x in pred_class]
    print(predictions)
    true_class = list(testcat)
    print(true_class)

    df = pd.DataFrame({'pred_class': pred_class, 'true_class': true_class})
    df = df.sort_values('pred_class', ascending=False)
    print(df.to_string())
    print(metrics.confusion_matrix(true_class, predictions))

    def eqArray(a, b):
        return np.where(a == b, 1, 0)

    a = np.sum(list(map(eqArray, predictions, true_class))) / len(true_class)
    return a
"""
res = Reservoir(n_internal_units=150, spectral_radius=0.6, leak=0.6,
                connectivity=0.01, input_scaling=0.1, noise_level=0.01,circle=False)

input_repr = res.getReservoirEmbedding(np.array(X), pca, ridge_embedding,n_drop=5, bidir=True, test=False)
print(input_repr)

input_repr_te = res.getReservoirEmbedding(np.array(sc_testdf), pca, ridge_embedding,n_drop=5, bidir=True, test=False)
print(input_repr_te)

readout.fit(input_repr, y)
pred_class = readout.predict(input_repr_te)
print(pred_class)
predictions = [int (round(x)) for x in pred_class]
print(predictions)
true_class = list(testcat)
print(true_class)

df = pd.DataFrame({'pred_class':pred_class, 'true_class':true_class})
df = df.sort_values('pred_class', ascending=False)
print(df.to_string())
print(metrics.confusion_matrix(true_class, predictions))

def eqArray(a,b):
    return np.where(a==b, 1, 0)

a = np.sum(list(map(eqArray, predictions, true_class))) / len(true_class)
print(a)
"""

pso = Pso(swarmsize=4, maxiter=14)
bp, value = pso.run(func,[50,0,0,0],[150,1,0.1,1])
v = func(bp)