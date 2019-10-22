import scipy.io 
import numpy as np
from numpy import linalg as LA

mat = scipy.io.loadmat('face.mat')

raw_data = mat['X']


raw_data = np.transpose(raw_data)

training_data = np.empty([int(520*0.8), 2576])
testing_data = np.empty([int(520*0.2), 2576])

print(training_data)
print(len(training_data))

for x in range(52):
	
	training_data[x*8:(x+1)*8] = raw_data[x*10:x*10+8]
	testing_data[x*2:(x+1)*2] = raw_data[x*10+8:(x+1)*10]

print(len(raw_data))
print(raw_data)
print("training_data")
print(len(training_data))
print(training_data)
print("testing_data")
print(len(testing_data))
print(testing_data)


cov_mat = np.cov(training_data)
print("length of covariance matrix")
print(len(cov_mat))

w, v = LA.eig(cov_mat)

print('w')
print(len(w))
print('v')
print(v)