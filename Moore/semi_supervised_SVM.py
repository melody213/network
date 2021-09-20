import pandas as pd
import numpy as np
import seaborn as sb
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
col_names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
             '21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40',
             '41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60',
             '61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80',
             '81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100',
             '101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116',
             '117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132',
             '133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148',
             '149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164',
             '165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180',
             '181','182','183','184','185','186','187','189','190','191','192','193','194','195','196','197',
             '198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213',
             '214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229',
             '230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245',
             '246','247','248','249','class']

df = pd.read_csv('data5.txt', names=col_names)
#print(df['class'].value_counts().sort_values(ascending=False))
"""
cols = df.select_dtypes(include=['float64', 'int64']).columns
values = df.loc[:, cols]
df1 = pd.DataFrame(values, columns=cols)
plt.figure(num=None, figsize=(60,60), dpi=80, facecolor='w', edgecolor='k')
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(df, hue='250')
plt.show()
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols = df.select_dtypes(include=['float64', 'int64']).columns
sc = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
sc_df = pd.DataFrame(sc, columns=cols)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cat = encoder.fit_transform(df['class'])
cat_df = pd.DataFrame(cat, columns=['class'])
df = pd.concat([sc_df, cat_df], axis=1)
test = df[df['class']==9].head(n=int(np.ceil(df[df['class']==9]).shape[0]*0.2))
test = test.append(df[df['class']==8].head(n=int(np.ceil(df[df['class']==8]).shape[0]*0.5)))
test = test.append(df[df['class']==7].head(n=int(np.ceil(df[df['class']==7]).shape[0]*0.5)))
test = test.append(df[df['class']==6].head(n=int(np.ceil(df[df['class']==6]).shape[0]*0.5)))
test = test.append(df[df['class']==5].head(n=int(np.ceil(df[df['class']==5]).shape[0]*0.5)))
test = test.append(df[df['class']==4].head(n=int(np.ceil(df[df['class']==4]).shape[0]*0.5)))
test = test.append(df[df['class']==3].head(n=int(np.ceil(df[df['class']==3]).shape[0]*0.5)))
test = test.append(df[df['class']==2].head(n=int(np.ceil(df[df['class']==2]).shape[0]*0.5)))
test = test.append(df[df['class']==1].head(n=int(np.ceil(df[df['class']==1]).shape[0]*0.5)))
test = test.append(df[df['class']==0].head(n=int(np.ceil(df[df['class']==0]).shape[0]*0.5)))

train = df.drop(test.index)
y_train = train['class']
print(y_train)
#training = train.drop('class',axis=1, inplace=True)
training = train[train.columns[:-1]]
print(training)
testing = test[test.columns[:-1]]
#testing = test.drop('class',axis=1, inplace=True)
"""
print(train[train.columns[-1]])
param_grid = {'C':[1, 0.1, 0.01, 0.001]}
#param_grid = {'C':range(0.1, 1)}
cv = KFold(5)
grid_search = GridSearchCV(LinearSVC(), param_grid, cv=cv)
grid_search.fit(train[train.columns[:-1]], train[train.columns[-1]])
print(grid_search.best_params_)

cclf = CalibratedClassifierCV(base_estimator=grid_search,cv='prefit')
cclf.fit(train[train.columns[:-1]], train[train.columns[-1]])

y_pred = grid_search.predict(test[test.columns[:-1]])
print(cclf.score(test[test.columns[:-1]], test['class']))
y_test = test['class']
y_train = train['class']

y_proba = cclf.predict_proba(test[test.columns[:-1]])[:,1]
fpr_rf, tpr_rf, _ = roc_curve(np.array(y_test), y_proba, pos_label=1)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label="AUC - "+str(auc(fpr_rf,tpr_rf)))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Test ROC curve')
plt.legend(loc='best')
plt.show()
print(roc_auc_score(test['class'], y_pred))

cm = confusion_matrix(test['class'], y_pred)
sns.heatmap(cm, annot=True, cbar=False, cmap='YlGnBu', fmt='d',)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(train['class'].value_counts().sort_values(ascending=False))
Labeled = train[train['class']==9].sample(frac=0.5)
Labeled = Labeled.append(train[train['class']==8].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==7].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==6].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==5].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==4].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==3].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==2].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==1].sample(frac=0.5))
Labeled = Labeled.append(train[train['class']==0].sample(frac=0.5))

print(Labeled['class'].value_counts())
print(Labeled.index)
Unlabeled = train.drop(Labeled.index)
print(Unlabeled['class'].value_counts())

param_grid = {'C':[1, 0.1, 0.01, 0.001]}
#param_grid = {'C':range(0.1, 1)}
cv = KFold(5)
grid_search = GridSearchCV(LinearSVC(penalty='l1', dual=False,max_iter=100), param_grid, cv=cv)
grid_search.fit(Labeled[Labeled.columns[:-1]], Labeled[Labeled.columns[-1]])
print(grid_search.best_params_)

while(Unlabeled.shape[0]!=0):
    clf=LinearSVC(penalty='l1', dual=False)
    clf.fit(Labeled[Labeled.columns[:-1]],Labeled['class'])
    distanceList=[]
    distanceList=abs(clf.decision_function(Unlabeled[Unlabeled.columns[:-1]].values))
    index = []
    for i in range(len(distanceList)):
        index.append(i)


    #index=sorted(index, key=lambda i: distanceList[i])[-1]

    toAdd = pd.DataFrame(Unlabeled.iloc[index]).T
    toAdd['class']=clf.predict(Unlabeled.iloc[index][Unlabeled.columns[:-1]])[0]
    Labeled=Labeled.append(toAdd)
    Unlabeled=Unlabeled.drop(Unlabeled.index[index])

clf.fit(Labeled[Labeled.columns[:-1]], Labeled['class'])
y_pred = clf.predict(test[test.columns[:-1]])
print(clf.score(train[test.columns[:1]], train['class']))

cclf = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
cclf.fit(Labeled[Labeled.columns[:-1]], Labeled['class'])
y_proba = cclf.predict_proba(test[test.columns[:-1]])[:,1]
"""
X_train, X_test = training, testing
features = training.columns
target = 'class'
num_folds = 10
from sklearn.linear_model import RidgeClassifier, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
"""
model_factory = [RandomForestClassifier(),
                 KNeighborsClassifier(),
                 ExtraTreesClassifier()]

for model in model_factory:
    num_folds = 10
    scores = cross_val_score(model, X_train, y_train,cv= num_folds,scoring='accuracy')
    cv_score = np.mean(scores)
    print(cv_score)
    print('{model:25} CV-5 Accuracy:{score}'.format(
        model=model.__class__.__name__,
        score=cv_score
    ))
"""
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
class PseudoLabeler(BaseEstimator, RegressorMixin):
    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        assert sample_rate <= 1.0
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''

        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
            augemented_train[self.features],
            augemented_train[self.target]
        )

        return self

    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])

        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__

model = PseudoLabeler(
    RandomForestClassifier(),
    test,
    features,
    target,
    sample_rate = 0.3
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
scores=cross_val_score(model, X_train, y_train, cv=num_folds, scoring='accuracy')
cv_score = np.mean(scores)
print(cv_score)

sample_rates = np.linspace(0, 1, 10)


def pseudo_label_wrapper(model):
    return PseudoLabeler(model, test, features, target)


# List of all models to test
model_factory = [RandomForestClassifier(),
                 KNeighborsClassifier(),
                 ExtraTreesClassifier()]
# Apply the PseudoLabeler class to each model
model_factory = map(pseudo_label_wrapper, model_factory)

# Train each model with different sample rates
results = {}

for model in model_factory:
    model_name = model.get_model_name()
    print('%s' % model_name)

    results[model_name] = list()
    for sample_rate in sample_rates:
        model.sample_rate = sample_rate

        # Calculate the CV-3 R2 score and store it
        scores = cross_val_score(model, X_train, y_train, cv=num_folds, scoring='accuracy', n_jobs=8)
        cv_score = np.mean(scores)
        results[model_name].append(cv_score)

plt.figure(figsize=(10, 5))

i = 1
for model_name, performance in results.items():
    plt.subplot(1, 3, i)
    i += 1

    plt.plot(sample_rates, performance)
    plt.title(model_name)
    plt.xlabel('sample_rate')
    plt.ylabel('accuracy')

plt.show()