import numpy as np
import scipy.io
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


class Reservoir(object):
    """
    Build a reservoir and evaluate internal states

    Parameters:
        n_internal_units = processing units in the reservoir
        spectral_radius = largest eigenvalue of the reservoir matrix of connection weights
        leak = amount of leakage in the reservoir state update (optional)
        connectivity = percentage of nonzero connection weights (unused in circle reservoir)
        input_scaling = scaling of the input connection weights
        noise_level = deviation of the Gaussian noise injected in the state update
        circle = generate determinisitc reservoir with circle topology
    """

    def __init__(self, n_internal_units=250, spectral_radius=0.99, leak=None,
                 connectivity=0.3, input_scaling=0.2, noise_level=0.01, circle=False):

        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._input_scaling = input_scaling
        self._noise_level = noise_level
        self._leak = leak

        # Input weights depend on input size: they are set when data is provided
        self._input_weights = None

        # Generate internal weights
        if circle:
            self._internal_weights = self._initialize_internal_weights_Circ(
                n_internal_units,
                spectral_radius)
        else:
            self._internal_weights = self._initialize_internal_weights(
                n_internal_units,
                connectivity,
                spectral_radius)

    def _initialize_internal_weights_Circ(self, n_internal_units, spectral_radius):

        internal_weights = np.zeros((n_internal_units, n_internal_units))
        internal_weights[0, -1] = spectral_radius
        for i in range(n_internal_units - 1):
            internal_weights[i + 1, i] = spectral_radius

        return internal_weights

    def _initialize_internal_weights(self, n_internal_units,
                                     connectivity, spectral_radius):

        # Generate sparse, uniformly distributed weights.

        internal_weights = sparse.rand(n_internal_units,
                                       n_internal_units,
                                       density=connectivity).todense()

        # Ensure that the nonzero values are uniformly distributed in [-0.5, 0.5]
        internal_weights[np.where(internal_weights > 0)] -= 0.5

        # Adjust the spectral radius.
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        internal_weights /= np.abs(e_max) / spectral_radius


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


df = pd.read_csv('data5.txt')
df1 = df[df['250']=='WWW'].head(n=int(df[df['250']=='WWW'].shape[0]*0.01))
print(df1)
print('\n')
df1 = df1.append(df[df['250']=='MAIL'].head(n=int(df[df['250']=='MAIL'].shape[0]*0.1)))
df1 = df1.append(df[df['250']=='FTP-CONTROL'].head(n=int(df[df['250']=='FTP-CONTROL'].shape[0]*0.1)))
df1 = df1.append(df[df['250']=='FTP-DATA'].head(n=int(df[df['250']=='FTP-DATA'].shape[0]*0.5)))
df1 = df1.append(df[df['250']=='ATTACK'])
df1 = df1.append(df[df['250']=='SERVICES'])
df1 = df1.append(df[df['250']=='P2P'])
df1 = df1.append(df[df['250']=='DATABASE'])
df1 = df1.append(df[df['250']=='MULTIMEDIA'])
df = df1.append(df[df['250']=='FTP-PASV'])

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
traincat = to_categorical(traincat)
testcat = to_categorical(testcat)
# 绘制流量分布图
df_train, df_test = train_test_split(df, test_size=0.5, random_state=0)
traffic_class_train = df_train[['250']].apply(lambda x:x.value_counts())
print(traffic_class_train)
traffic_class_test = df_test[['250']].apply(lambda x:x.value_counts())
print(traffic_class_test)
traffic_class_dist = pd.concat([traffic_class_train, traffic_class_test], axis=1)
print(traffic_class_dist)
plt.figure(figsize=(10, 5))
plot = traffic_class_dist.plot(kind='bar')
plot.set_title("Traffic Class", fontsize=20)
plot.grid(color='lightgray', alpha=0.5)
plt.legend(['train data', 'test data'])
plt.show()



#enctrain = pd.DataFrame(traincat, columns=['250'])
enctrain = pd.DataFrame(traincat, columns=['250', '251', '252', '253', '254', '255', '256', '257', '258', '259'])
refclasscol = pd.concat([sc_traindf, enctrain], axis=1).columns
refclass = np.concatenate((sc_traindf.values, enctrain.values), axis=1)
X = refclass
"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
X = sc_traindf
y = traincat
rfc.fit(X, y)
score = np.round(rfc.feature_importances_,3)
importances = pd.DataFrame({'feature': cols, 'importance':score})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
plt.rcParams['figure.figsize'] = (11, 4)
importances.plot.bar()
plt.show()


"""
from sklearn.ensemble import RandomForestClassifier
samplecols = sc_traindf.columns.values
X = sc_traindf
y = traincat
rfc = RandomForestClassifier(random_state=123,criterion='entropy')
rfc.fit(X, y)
score = np.argsort(rfc.feature_importances_)[::-1]
feature_rank = pd.DataFrame(columns=['rank', 'feature', 'importance'])
for f in range(sc_traindf.shape[1]):
    feature_rank.loc[f] = [f+1,
                           sc_traindf.columns[score[f]],
                           rfc.feature_importances_[score[f]]]

importances = pd.DataFrame({'feature': cols, 'importance':score})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
plt.rcParams['figure.figsize'] = (11, 4)
importances.plot.bar()
plt.show()

top30 = list(feature_rank.feature[0:30])
print(top30)
"""
X = sc_traindf[top30]
print(X)
"""

X = np.reshape(X.values, (X.values.shape[0], X.values.shape[1], 1))
sc_testdf = np.reshape(sc_testdf.values, (sc_testdf.values.shape[0], sc_testdf.values.shape[1], 1))

pca = PCA(n_components=100)
ridge_embedding = Ridge(max_iter=5)
readout = Ridge(max_iter=5)

res = Reservoir(n_internal_units=250, spectral_radius=1, leak=1,
                 connectivity=1, input_scaling=1, noise_level=0.01, circle=False)

input_repr = res.getReservoirEmbedding(X, pca, ridge_embedding,  n_drop=5, bidir=True, test = False)
print(input_repr)
input_repr_te = res.getReservoirEmbedding(sc_testdf, pca, ridge_embedding,  n_drop=5, bidir=True, test = True)
print(input_repr_te)
readout.fit(input_repr, traincat)
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
plt.figure(figsize=(15, 10))
plt.imshow(confusion, cmap=plt.cm.Blues)
for first_index in range(len(confusion)):    #第几行
    for second_index in range(len(confusion[first_index])):    #第几列
        plt.text(first_index, second_index, confusion[first_index][second_index], va='center', ha='center')

#indices = range(len(confusion))
#plt.xticks(indices, [0, 1])
#plt.yticks(indices, [1, 0])
plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('ESN混淆矩阵')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
indices = range(len(confusion))
plt.xticks(indices, ['ATTACK', 'DATABASE', 'FTP-CONTROL', 'FTP-DATA', 'FTP-PASV',
                     'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW'])
plt.yticks(indices, ['ATTACK', 'DATABASE', 'FTP-CONTROL', 'FTP-DATA', 'FTP-PASV',
                     'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW'])
plt.show()
"""

input_repr = res.getReservoirEmbedding(np.array(X), pca, ridge_embedding,n_drop=5, bidir=True, test=False)
print(input_repr)

input_repr_te = res.getReservoirEmbedding(np.array(sc_testdf), pca, ridge_embedding, n_drop=5, bidir=True, test=False)
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
