import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score,f1_score,recall_score,precision_score
from sklearn import datasets
import scipy.stats as stats
import keras
# Network Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from sklearn import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint

#Import the network optimizers
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta, Adamax, SGD

#Import utilities
import numpy as np
from keras.utils.np_utils import to_categorical
from operator import add
from functools import reduce
from keras import backend as K
import random

df = pd.read_csv('data_set5.txt')
X = df.iloc[:, [0, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
Y = df.iloc[:, -1]
columns = ['flow_duration_sec', 'flow_duration_nsec', 'idle_timeout', 'hard_timeout', 'packet_count',
           'byte_count', 'packet_count_per_second', 'packet_count_per_nsecond', 'byte_count_per_second', 'byte_count_per_nsecond']
from sklearn.preprocessing import MinMaxScaler
columns_values = X[columns]
sc = MinMaxScaler(feature_range=(0,1))
sc1 = sc.fit_transform(columns_values)
X = pd.DataFrame(sc1, columns=columns)
X_newdf = int(len(X) * 0.8)
X_newdf_test = len(X) - X_newdf
print(X_newdf, X_newdf_test)
X_newdf = X[0:X_newdf]
print(X_newdf)
X_newdf_test = X[len(X_newdf):]

Y_newdf = int(len(Y) * 0.8)
Y_newdf_test = len(Y) - Y_newdf
print(Y_newdf, Y_newdf_test)
Y_newdf = Y[0:Y_newdf]
print(Y_newdf)
Y_newdf_test = Y[len(Y_newdf):]


X_newdf = np.reshape(X_newdf.values, (X_newdf.values.shape[0], X_newdf.values.shape[1], 1))
X_newdf_test = np.reshape(X_newdf_test.values, (X_newdf_test.values.shape[0], X_newdf_test.values.shape[1], 1))

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


iter_max = 5
pop_size = 20
c1 = 2.5
c2 = 2
w = 1.7

hyperPar = []
for i in range(pop_size):
    particle = {'CNN':[np.random.randint(1,8),np.random.randint(0,8),np.random.randint(0,8),np.random.uniform(0,0.5)]}
    hyperPar.append(particle)

swarm = []
for i in range(pop_size):
    p = {}
    p[0] = hyperPar[i]
    p[1] = 999999999
    p[2] = 0.0
    p[3] = p[0]
    swarm.append(p)
out2 = open('PSO_AmazonBestParticle.txt','w')
out2.write("Params:\n"+"c1: "+str(c1)+"\n"+"c2: "+str(c2)+"\n"+"w: "+str(w)+"\n"+"swarm size: "+str(pop_size)+"\n"+"iterations: "+str(iter_max)+"\n")
out = open("PSO_Amazon_trace.csv", "w")
out.write("Iteration,LogLoss\n")
print(swarm[0])
j=0
gbest = swarm[0]
while j<iter_max:
    print('------->' + str(j))
    for p in swarm:
        model = Sequential()
        # Add Convolutional layers

        model.add(Convolution1D(filters=2 ** p[0]['CNN'][0], kernel_size=3, activation='relu', padding='same',input_shape=(10, 1)))
        model.add(MaxPooling1D(pool_size=2))

        if p[0]['CNN'][1] > 0:
            model.add(Convolution1D(filters=2**p[0]['CNN'][1], kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))

        if p[0]['CNN'][2] > 0:
            model.add(Convolution1D(filters=2 ** p[0]['CNN'][2], kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))

        model.add(Dropout(p[0]['CNN'][3]))
        model.add(Flatten())
        # Densely connected layers
        model.add(Dense(128, activation='relu'))
        # Output layer
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        batch_size = 32
        num_classes = 2
        epochs = 10
        history = model.fit(X_newdf, Y_newdf, batch_size=batch_size, epochs=epochs, verbose=1)
        predictions = model.predict(X_newdf_test)
        fitness = log_loss(Y_newdf_test, predictions)
        Y_newdf_pred = model.predict_classes(X_newdf_test)
        accuracy = accuracy_score(Y_newdf_test, Y_newdf_pred)
        recall = recall_score(Y_newdf_test, Y_newdf_pred, average="binary")
        precision = precision_score(Y_newdf_test, Y_newdf_pred, average="binary")
        f1 = f1_score(Y_newdf_test, Y_newdf_pred, average="binary")

        print("confusion matrix")
        print("----------------------------------------------")
        print("accuracy")
        print("%.6f" % accuracy)
        print("recall")
        print("%.6f" % recall)
        print("precision")
        print("%.6f" % precision)
        print("f1score")
        print("%.6f" % f1)

        cm = metrics.confusion_matrix(Y_newdf_test, Y_newdf_pred)
        print(cm)
        print("==============================================")
        print(fitness)
        if fitness<gbest[1]:
            print('\n*** Global Best! '+str(fitness)+"\n")
            gbest = p
            gbest[3] = p[0]

        if fitness < p[1]:
            print("--- Local Best! "+str(fitness))
            p[1]= fitness
            p[3] = p[0]
        else:
            p[1] = fitness

        for clf in p[0].keys():
            if clf == 'CNN':
                v = w*p[2]+ c1 * np.random.uniform(0,1) * (p[3][clf][0] - p[0][clf][0]) + c2 * np.random.uniform(0,1) * (gbest[3][clf][0] - p[3][clf][0])
                p[0][clf][0] = abs(p[0][clf][0] + round(v))
                v = round(w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf][1] - p[0][clf][1]) + c2 * np.random.uniform(0, 1) * (gbest[3][clf][1] - p[3][clf][1]))
                p[0][clf][1] = abs(p[0][clf][1] + v)
                v = round(w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf][2] - p[0][clf][2]) + c2 * np.random.uniform(0,1) * (gbest[3][clf][2] - p[3][clf][2]))
                p[0][clf][2] = abs(p[0][clf][2] + v)
                v = round(w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf][3] - p[0][clf][3]) + c2 * np.random.uniform(0,1) * (gbest[3][clf][3] - p[3][clf][3]))
                p[0][clf][3] = abs(p[0][clf][3] + v)






    out.write(str(j + 1) + "," + str(gbest[1]) + "\n")
    j += 1

for clf in gbest[0].keys():
    out2.write(clf+'\t'+str(gbest[0][clf]+'\n'))


