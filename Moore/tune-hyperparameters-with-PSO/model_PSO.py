import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,BaggingClassifier,VotingClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import inspect

df = pd.read_csv('data5.txt')

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
traffic_class_train = df_train[['250']].apply(lambda x:x.value_counts())
print(traffic_class_train)
traffic_class_test = df_test[['250']].apply(lambda x:x.value_counts())
print(traffic_class_test)
traffic_class_dist = pd.concat([traffic_class_train, traffic_class_test], axis=1)
print(traffic_class_dist)
"""
plt.figure(figsize=(10, 5))
plot = traffic_class_dist.plot(kind='bar')
plot.set_title("Traffic Class", fontsize=20)
plot.grid(color='lightgray', alpha=0.5)
plt.legend(['train data', 'test data'])
plt.show()
"""


enctrain = pd.DataFrame(traincat, columns=['250'])
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

iter_max = 20
pop_size = 100
c1 = 2.5
c2 = 2
w = 1.7

def errorFunction(classifier):
    classifierFit = classifier.fit(sc_traindf, traincat)
    predictions = classifierFit.predict_proba(sc_testdf)
    score = log_loss(testcat, predictions)
    return score

hyperPar = []
for i in range(pop_size):
    particle = {'RF':np.random.randint(100, 500), 'SVM':np.random.uniform(0.1,7.0), 'LogReg':[np.random.uniform(0.1,7.0),np.random.randint(10,1000)], 'KNN':np.random.randint(10,100)}
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
        rf = RandomForestClassifier(n_estimators=p[0]['RF'], bootstrap=True)
        svm = SVC(kernel='poly',gamma='auto', C=p[0]['SVM'], probability=True)
        lg = LogisticRegression(solver='newton-cg',max_iter = p[0]['LogReg'][1],C =p[0]['LogReg'][0])
        knn = KNeighborsClassifier(n_neighbors=p[0]['KNN'])
        lda = LinearDiscriminantAnalysis(solver='svd')
        vote = VotingClassifier(estimators=[('SVM', svm), ('Random Forests', rf), ('LogReg', lg), ('KNN', knn), ('LDA',lda)], voting='soft')
        fitness = errorFunction(vote)
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
            if clf == 'LogReg':
                v = w*p[2]+ c1 * np.random.uniform(0,1) * (p[3][clf][0] - p[0][clf][0]) + c2 * np.random.uniform(0,1) * (gbest[3][clf][0] - p[3][clf][0])
                p[0][clf][0] = abs(p[0][clf][0] + round(v))
                v = round(w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf][1] - p[0][clf][1]) + c2 * np.random.uniform(0, 1) * (gbest[3][clf][1] - p[3][clf][1]))
                p[0][clf][1] = abs(p[0][clf][1] + v)

            elif clf == 'RF':
                v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0, 1) * (gbest[3][clf] - p[3][clf])
                p[0][clf] = abs(p[0][clf] + round(v))

            elif clf == 'KNN':  # n_neighbors (KNN)
                v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0, 1) * (gbest[3][clf] - p[3][clf])
                newPosition = abs(p[0][clf] + round(v))
                if newPosition == 0:
                    print("N neighbours = 0")
                    p[0][clf] = np.random.randint(5, 50)
                else:
                    p[0][clf] = newPosition

            else:  # C (SVM)
                v = w * p[2] + c1 * np.random.uniform(0, 1) * (p[3][clf] - p[0][clf]) + c2 * np.random.uniform(0, 1) * (gbest[3][clf] - p[3][clf])
                p[0][clf] = abs(p[0][clf] + v)

    out.write(str(j + 1) + "," + str(gbest[1]) + "\n")
    j += 1

for clf in gbest[0].keys():
    out2.write(clf+'\t'+str(gbest[0][clf]+'\n'))




