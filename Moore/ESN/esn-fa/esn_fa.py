import numpy as np
from random import random,uniform,choice,randint
from time import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import sys
from keras.utils.np_utils import to_categorical

num_digits = 3

class GA:
    def __init__(self, fitness, parameters, populationSize, maxGen,
                 retain=0.2, random_select = 0.05, mutate = 0.01):
        self.fitness = fitness
        self.parametersRange = list(parameters.values())
        self.populationSize = populationSize
        self.maxGen = maxGen
        self.retain = retain
        self.random_select = random_select
        self.mutate = mutate
        self.history = {}
        self.stop = False

    def individual(self):
        return [round(uniform(*parameter), num_digits) if type(parameter) == tuple else choice(parameter)
                for parameter in self.parametersRange]

    def population(self,count):
        pop = []
        while len(pop)<count:
            ind = self.individual()
            if ind not in pop:
                pop.append(ind)
        return pop

    def grade(self,list_fit=None):
        if not list_fit:
            list_fit = self.fit

        try:
            return np.nanmin(fit for fit in self.fit)
        except:
            return np.nan

    def evaluate_individual(self, individual):
        individual_tuple = tuple(individual)
        if individual_tuple not in self.history:
            fitness = self.fitness(individual)
            if np.isnan(fitness):
                fitness = self.fitness(individual)

            self.history[individual_tuple]=fitness
        return self.history[individual_tuple]

    def evaluate_population(self,population):
        return [self.evaluate_individual(individual) for individual in tqdm(population,desc='  Evaluating',file=sys.stdout)]

    def evolve(self):
        retain_length = int(self.populationSize*self.retain)
        parents = [ind for ind in tqdm(self.pop[:retain_length], desc="    Retain", file=sys.stdout)]
        for individual in tqdm(self.pop[retain_length:], desc="    Random", file=sys.stdout):
            if self.random_select>random():
                parents.append(individual)

        for individual in tqdm(parents,desc="    Mutation", file=sys.stdout):
            if self.mutate>random():
                pos_to_mutate = randint(0, len(individual) - 1)
                #pos_to_mutate = uniform(0, len(individual)-1)
                parameter = self.parametersRange[pos_to_mutate]
                individual[pos_to_mutate] = uniform(*parameter) if type(parameter) == tuple else choice(parameter)

        desired_length = self.populationSize-len(parents)
        unique = np.unique(self.pop, axis=0)
        if (len(unique) < 2):
            # self.stop = True
            print("  # of different elements < 2")
            extend = [self.pop[0]] * desired_length
            parents.extend(extend)
        else:
            # crossover parents to create children
            children = []
            cont = 0
            while len(children) < desired_length:
                male, female = choice(parents), choice(parents)
                cont = 0
                while male == female:
                    cont += 1
                    female = choice(parents)
                    if cont > 1000:
                        female = self.individual()
                position = randint(1, len(male) - 1)
                #position = uniform(1, len(male) - 1)
                child1 = male[:position] + female[position:]
                child2 = female[:position] + male[position:]
                children.append(child1)
                children.append(child2)
            parents.extend(children)
            print("    Crossover: OK")

        # organizing the new population
        parents = sorted(parents[:self.populationSize])
        print("    New Population:", parents)

        # evaluating new population
        new_fit = self.evaluate_population(parents)

        new_best = self.grade(new_fit)
        print("    Best fitness of this generation:", new_best)

        # # if the new best is bigger than the old one by 60% or higher, stop.
        # if((new_best - self.best_fit)/self.best_fit >= 0.6):
        #   print("  Stopping due to: moving out from convergence")
        #   self.stop = True
        #   return self.best_fit

        # sorting the new population by fitness
        self.fit, self.pop = [list(t) for t in zip(*sorted(zip(new_fit, parents)))]
        self.best_fit = new_best

        # # check if GA has already converged
        # # if number of different elements <= 20%, stop.
        # unique = np.unique(self.pop, axis=0)
        # if(len(unique) <= (0.2 * self.populationSize)):
        #   self.stop = True
        #   print("  Stopping due to: # different elements <= 20%")

        # elif(self.best_fit <= 1e-3):
        #   print("  Stopping due to: convergence")
        #   self.stop = True

        # elif(np.isnan(self.best_fit)):
        #   print("  Stopping due to: best fitness is NaN")
        #   self.stop = True

        return self.best_fit

    def report(self, grade_history):
        best_index = self.fit.index(min(self.fit))
        best = {'gene': self.pop[best_index], 'loss': self.fit[best_index]}
        return best, grade_history

    def plot(self, hist, op="show", name=None, lang='en'):
        x = [i + 1 for i in range(len(hist))]
        y = hist
        # English
        plt.plot(x, y)
        plt.xlabel("# of iterations" if lang == 'en' else "Número de iterações")
        plt.ylabel("Loss")
        if op == "show":
            plt.show()
        else:
            plt.savefig(name)
        plt.gcf().clear()


    def run(self):
        self.history = {}
        self.pop = sorted(self.population(self.populationSize))
        print("    population:", self.pop)
        self.fit = self.evaluate_population(self.pop)
        self.best_fit = self.grade()
        print('    Initial best:', self.best_fit)
        self.fit, self.pop = [list(t) for t in zip(*sorted(zip(self.fit,self.pop)))]
        grade_history = []
        for i in range(self.maxGen):
            print(f'\n Running generation {(i+1)}/{self.maxGen}')
            t = time()
            grade = self.evolve()
            grade_history.append(grade)
            print('   End of generation.\n Time elapsed (s):', time() -t)
        return self.report(grade_history)

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
    traincat = to_categorical(traincat)
    testcat = to_categorical(testcat)

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
    enctrain = pd.DataFrame(traincat,
                            columns=['250', '251', '252', '253', '254', '255', '256', '257', '258', '259'])
    refclasscol = pd.concat([sc_traindf, enctrain], axis=1).columns
    refclass = np.concatenate((sc_traindf.values, enctrain.values), axis=1)
    X = refclass

    X = sc_traindf
    y = traincat
    X = np.reshape(X.values, (X.values.shape[0], X.values.shape[1], 1))
    sc_testdf = np.reshape(sc_testdf.values, (sc_testdf.values.shape[0], sc_testdf.values.shape[1], 1))

    n, m, e, f = x[0], x[1], x[2], x[3]
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

    def eqArray(a, b):
        return np.where(a == b, 1, 0)

    a = np.sum(list(map(eqArray, predict_target2, testcat))) / len(testcat)
    return 1-a

from collections import OrderedDict
# from keras.optimizers import Adam, Adadelta, Adagrad, Adamax, SGD, RMSprop

params = OrderedDict([
    ('n_internal_units', np.arange(20, 50)),
    ('spectral_radius', np.arange(0.4, 1,0.1)),
    ('connectivity', np.arange(0.4, 1,0.1)),
    ('input_scaling', np.arange(0.4, 1,0.1))
])

def run(num_executions=5):
    results = []
    loss = []
    for i in range(num_executions):
      print(f"\nRunning execution {(i+1)}/{num_executions}")
      # Run Evolver
      best, hist = evolver.run()
      print('BEST GENE', best['gene'])
      # Calculate loss
      gen_loss = evolver.fitness(best['gene'])
      print('gen_loss', gen_loss)
      """
      loss.append(gen_loss)
      # Store results
      results.append({
        'best': best,
        'gen_loss': gen_loss,
        'hist': hist,
        'pop': evolver.pop,
        'fit': evolver.fit,
        'history': evolver.history
      })
      """
    return gen_loss

evolver = GA(func, params, populationSize=5, maxGen=5)
loss = run()

