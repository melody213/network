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
rfc = RandomForestClassifier(random_state=123, criterion='entropy')
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
from sklearn.svm import SVC
from sklearn import metrics

#model = SVC(gamma='auto')
model = SVC()
model.fit(X, y)
predict_target = model.predict(X)
print("训练集：")
print("预测正确数量，训练集样本量：")
print(sum(predict_target == y), len(y))
print("精确度等指标")
print(metrics.classification_report(y, predict_target))
print("混淆矩阵")
print(metrics.confusion_matrix(y, predict_target))
print("测试集：")

print("测试集：")
#predict_target2 = model.predict(sc_testdf[top30])
predict_target2 = model.predict(sc_testdf)
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
plt.title('支持向量机混淆矩阵')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
indices = range(len(confusion))
plt.xticks(indices, ['DNS', 'FTP', 'HTTP', 'TELNET', 'lime',
                     'localForwarding', 'remoteForwarding', 'scp', 'sftp', 'shell', 'x11'])
plt.yticks(indices, ['DNS', 'FTP', 'HTTP', 'TELNET', 'lime',
                     'localForwarding', 'remoteForwarding', 'scp', 'sftp', 'shell', 'x11'])
plt.show()