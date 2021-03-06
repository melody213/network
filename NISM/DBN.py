from sklearn.neural_network import BernoulliRBM
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical

class DBN():

    def __init__(
            self,
            train_data,
            targets,
            layers,
            outputs,
            rbm_lr,
            rbm_iters,
            rbm_dir=None,
            test_data=None,
            test_targets=None,
            epochs=25,
            fine_tune_batch_size=32
    ):

        self.hidden_sizes = layers
        self.outputs = outputs
        self.targets = targets
        self.data = train_data

        if test_data is None:
            self.validate = False
        else:
            self.validate = True

        self.valid_data = test_data
        self.valid_labels = test_targets

        self.rbm_learning_rate = rbm_lr
        self.rbm_iters = rbm_iters

        self.epochs = epochs
        self.nn_batch_size = fine_tune_batch_size

        self.rbm_weights = []
        self.rbm_biases = []
        self.rbm_h_act = []

        self.model = None
        self.history = None

    def pretrain(self, save=True):

        visual_layer = self.data

        for i in range(len(self.hidden_sizes)):
            print("[DBN] Layer {} Pre-Training".format(i + 1))

            rbm = BernoulliRBM(n_components=self.hidden_sizes[i], n_iter=self.rbm_iters[i],
                               learning_rate=self.rbm_learning_rate[i], verbose=True, batch_size=32)
            rbm.fit(visual_layer)
            self.rbm_weights.append(rbm.components_)
            self.rbm_biases.append(rbm.intercept_hidden_)
            self.rbm_h_act.append(rbm.transform(visual_layer))

            visual_layer = self.rbm_h_act[-1]

    def finetune(self):
        model = Sequential()
        for i in range(len(self.hidden_sizes)):

            if i == 0:
                model.add(Dense(self.hidden_sizes[i], activation='relu', input_dim=self.data.shape[1],
                                name='rbm_{}'.format(i)))
            else:
                model.add(Dense(self.hidden_sizes[i], activation='relu', name='rbm_{}'.format(i)))

        model.add(Dense(self.outputs, activation='softmax'))
        model.compile(optimizer='Adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        for i in range(len(self.hidden_sizes)):
            layer = model.get_layer('rbm_{}'.format(i))
            layer.set_weights([self.rbm_weights[i].transpose(), self.rbm_biases[i]])

        if self.validate:
            self.history = model.fit(sc_traindf, traincat,
                                     epochs=self.epochs,
                                     batch_size=self.nn_batch_size,
                                     verbose=1)
            predict_target2 = model.predict(sc_testdf)
            predict_target2 = np.argmax(predict_target2, axis=1)

            testcat = np.argmax(self.valid_labels, axis=1)

            from sklearn import metrics

            print(predict_target2)
            print("??????????????????????????????????????????")
            print(sum(predict_target2 == testcat), len(testcat))
            print("?????????????????????")
            print(metrics.classification_report(testcat, predict_target2))
            print("???????????????")
            print(metrics.confusion_matrix(testcat, predict_target2))

            confusion = metrics.confusion_matrix(testcat, predict_target2)
            plt.figure(figsize=(15, 10))
            plt.imshow(confusion, cmap=plt.cm.Blues)
            for first_index in range(len(confusion)):  # ?????????
                for second_index in range(len(confusion[first_index])):  # ?????????
                    plt.text(first_index, second_index, confusion[first_index][second_index], va='center', ha='center')

            # indices = range(len(confusion))
            # plt.xticks(indices, [0, 1])
            # plt.yticks(indices, [1, 0])
            plt.xlabel('?????????')
            plt.ylabel('?????????')
            plt.title('DBN????????????')
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            indices = range(len(confusion))
            plt.xticks(indices, ['ATTACK', 'DATABASE', 'FTP-CONTROL', 'FTP-DATA', 'FTP-PASV',
                                 'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW'])
            plt.yticks(indices, ['ATTACK', 'DATABASE', 'FTP-CONTROL', 'FTP-DATA', 'FTP-PASV',
                                 'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW'])
            plt.show()


        self.model = model


num_classes = 11

df = pd.read_csv('NISM.csv')
df1 = df[df['label']=='DNS'].head(n=int(df[df['label']=='DNS'].shape[0]*0.1))
print(df1)
print('\n')
df1 = df1.append(df[df['label']=='lime'].head(n=int(df[df['label']=='lime'].shape[0]*0.006)))
df1 = df1.append(df[df['label']=='FTP'])
df1 = df1.append(df[df['label']=='HTTP'].head(n=int(df[df['label']=='HTTP'].shape[0]*0.2)))
df1 = df1.append(df[df['label']=='TELNET'])
df1 = df1.append(df[df['label']=='localForwarding'])
df1 = df1.append(df[df['label']=='remoteForwarding'])
df1 = df1.append(df[df['label']=='scp'])
df1 = df1.append(df[df['label']=='sftp'])
df1 = df1.append(df[df['label']=='shell'])
df = df1.append(df[df['label']=='x11'])
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
# ?????????????????????
df_train, df_test = train_test_split(df, test_size=0.5, random_state=0)
traffic_class_train = df_train[['label']].apply(lambda x:x.value_counts())
print(traffic_class_train)
traffic_class_test = df_test[['label']].apply(lambda x:x.value_counts())
print(traffic_class_test)
traffic_class_dist = pd.concat([traffic_class_train, traffic_class_test], axis=1)
print(traffic_class_dist)
plt.figure(figsize=(10, 5))
plot = traffic_class_dist.plot(kind='bar')
plot.set_title("Traffic Class", fontsize=20)
plot.grid(color='lightgray', alpha=0.5)
plt.legend(['train data', 'test data'])
plt.show()

enctrain = pd.DataFrame(traincat, columns=['250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260'])

refclasscol = pd.concat([sc_traindf, enctrain], axis=1).columns
refclass = np.concatenate((sc_traindf.values, enctrain.values), axis=1)
X = refclass
if __name__ == '__main__':
  dbn = DBN(train_data = sc_traindf, targets = traincat,
            test_data = sc_testdf, test_targets = testcat,
            layers = [200],
            outputs = 11,
            rbm_iters = [40],
            rbm_lr = [0.01])
  dbn.pretrain(save=True)
  dbn.finetune()