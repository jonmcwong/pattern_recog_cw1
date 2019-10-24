import scipy.io 
import numpy as np
from numpy import linalg as LA
import matplotlib
import time

# load MATLAB matrix into a python dictionary
mat = scipy.io.loadmat('face.mat')

# lookup the data within the dictionary object
# There are headers so we can't use the dictionary as a data only object
raw_data = mat['X']

#transpose to the correct orientation so we can separate the images into training and testing datasets
raw_data = np.transpose(raw_data)

# init arrays for training and test data
training_data = np.empty([int(520*0.8), 2576])
testing_data = np.empty([int(520*0.2), 2576])


# create training and test data
for x in range(52):
        # 8/2 ratio for training and testing datasets	
	training_data[x*8:(x+1)*8] = raw_data[x*10:x*10+8]
	testing_data[x*2:(x+1)*2] = raw_data[x*10+8:(x+1)*10]

#compute the covariance matrix for low-dimensional comutation
start = time.time()
low_cov_mat = np.cov(training_data)
end = time.time()
print("time required to compute low-dim-cov-mat : ",end - start)
print("length of covariance matrix")
print(len(low_cov_mat))

# compute the eigenvalues and eigenvectors of the covariance matrix
# w -> eigenvalues
# v -> eigenvectors
start = time.time()
low_w, low_v = LA.eig(low_cov_mat)

# reconstruct original eignevectors
normal_u = training_data.T.dot(low_v) # is probably incredibly wrong
end = time.time()
print("time required to compute eigenvalues and eigenvectors of low-dim-cov-mat : ",end-start)

# compute the covariance matrix for normal PCA computation
normal_cov_mat = np.cov(np.transpose(training_data))
print("Length of normal cov matrix")
print(len(normal_cov_mat))

# compute eigenvalues and vectors for normal PCA computation




