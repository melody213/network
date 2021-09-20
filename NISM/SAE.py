import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

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

# 绘制流量分布图
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

enctrain = pd.DataFrame(traincat, columns=['label'])
refclasscol = pd.concat([sc_traindf, enctrain], axis=1).columns
refclass = np.concatenate((sc_traindf.values, enctrain.values), axis=1)
X = refclass

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.models import Model

"""

input_data = Input(shape=(216, ))
crrpt_data = Dropout(0.5)(input_data)
encoded = Dense(250, activation='sigmoid')(crrpt_data)
decoded = Dense(216, activation='relu')(encoded)

autoencoder = Model(input_data, decoded)

nb_epoch = 10
batch_size = 32
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = autoencoder.fit(sc_traindf, sc_traindf, epochs=nb_epoch, batch_size=batch_size,
                          shuffle=True,  verbose=1)

encoder = Model(input_data, encoded)
predict_target2 = encoder.predict(sc_testdf)




from sklearn import metrics

print(predict_target2)
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
plt.title('神经网络混淆矩阵')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
indices = range(len(confusion))
plt.xticks(indices, ['ATTACK', 'DATABASE', 'FTP-CONTROL', 'FTP-DATA', 'FTP-PASV',
                     'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW'])
plt.yticks(indices, ['ATTACK', 'DATABASE', 'FTP-CONTROL', 'FTP-DATA', 'FTP-PASV',
                     'MAIL', 'MULTIMEDIA', 'P2P', 'SERVICES', 'WWW'])
plt.show()
"""

encoding_dim = 32
input_data = Input(shape=(22, ))
crrpt_data = Dropout(0.5)(input_data)
encoded = Dense(250, activation='relu')(crrpt_data)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(1, activation='relu')(decoded)

autoencoder = Model(input_data, decoded)
encoder = Model(input_data, encoded)
encoded_input = Input(shape=(encoding_dim,))
deco = autoencoder.layers[-3](encoded_input)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)
nb_epoch = 10
batch_size = 32
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = autoencoder.fit(sc_traindf, traincat, epochs=nb_epoch, batch_size=batch_size,
                          shuffle=True,  verbose=1)

encoded_data = encoder.predict(sc_testdf)
decoded_data = decoder.predict(encoded_data)

predict_target2 = decoded_data
for i in range(len(predict_target2)):
    if (predict_target2[i] - int(predict_target2[i])) > 0.5:
        predict_target2[i] = int(predict_target2[i]) + 1

    else:
        predict_target2[i] = int(predict_target2[i])
    print(predict_target2[i])

from sklearn import metrics

print(predict_target2)
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
plt.title('SAE混淆矩阵')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
indices = range(len(confusion))
plt.xticks(indices, ['DNS', 'FTP', 'HTTP', 'TELNET', 'lime',
                     'localForwarding', 'remoteForwarding', 'scp', 'sftp', 'shell', 'x11'])
plt.yticks(indices, ['DNS', 'FTP', 'HTTP', 'TELNET', 'lime',
                     'localForwarding', 'remoteForwarding', 'scp', 'sftp', 'shell', 'x11'])
plt.show()
