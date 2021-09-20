import numpy as np
NF = 5
k = 3
F=np.zeros((NF,k,2))
print(F)

for i in range(NF):
    C_x = np.random.randint(50, 200, size=k)
    print(C_x)
    C_y = np.random.randint(0, 50, size=k)
    print(C_y)
    Y = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    F[i]=Y

print("Initial firefly")
print(F)

E = np.zeros((5, 2))
print(E)
C_x = np.random.randint(20, 50, 5)
print(C_x)
C_y = np.random.rand(5)
print(C_y)
C_z = np.random.rand(5)
C_w = np.random.rand(5)
Y = np.array(list(zip(C_x, C_y, C_z, C_w)))
    #E[i] = Y
print(Y)

Positions = np.dot(np.random.rand(20,2),(10-0.01))+0.01
print(Positions)

SearchAgents_no = 20  # 狼群数量
Max_iteration = 20  # 最大迭代次数
dim = 4  # 需要优化两个参数c和g
rnges = {'n_internal_units_lower': 20, 'n_internal_units_upper': 50, 'spectral_radius_lower': 0,
        'spectral_radius_upper': 1,
        'connectivity_lower': 0, 'connectivity_upper': 1, 'input_scaling_lower': 0, 'input_scaling_upper': 1}


spectral_radius_range = rnges['spectral_radius_upper'] - rnges['spectral_radius_lower']
connectivity_range = rnges['connectivity_upper'] - rnges['connectivity_lower']
input_scaling_range = rnges['input_scaling_upper'] - rnges['input_scaling_lower']
xn = np.random.randint(low=rnges['n_internal_units_lower'], high=rnges['n_internal_units_upper'],
                        size=SearchAgents_no)
yn = np.random.rand(SearchAgents_no) * spectral_radius_range + rnges['spectral_radius_lower']
zn = np.random.rand(SearchAgents_no) * connectivity_range + rnges['connectivity_lower']
wn = np.random.rand(SearchAgents_no) * input_scaling_range + rnges['input_scaling_lower']

Positions = np.array(list(zip(xn, yn, zn, wn)))
print(Positions)
print(Positions.shape[0])
print(Positions[1,3])
print(Positions[1][0])
print(Positions[0,-1])

position = np.zeros((5, 3))
print(position)
a = position[0, 0:position.shape[1] - 1]
print(a)
