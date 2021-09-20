import random
def set_random_parameters(parameter_space):
    for parameter in parameter_space:
        parameter = random.choice(parameter_space[parameter])
parameter_space = {
    'optimizer': ['Adam', 'rmsprop', 'Adagrad', 'Adadelta', 'Adamax', 'SGD'],
    'layer_1': [0, 1, 2, 3, 4, 5, 6, 7],
    'layer_2': [0, 1, 2, 3, 4, 5, 6, 7],
    'layer_3': [0, 1, 2, 3, 4, 5, 6, 7],
    'Dropout': [0, 0.25, 0.5],
    'learning_rate': [0.1, 0.01, 0.001, 0.0001]
}

parameter = set_random_parameters(parameter_space)
print(parameter)