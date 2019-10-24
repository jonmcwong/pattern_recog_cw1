import scipy.io 
import numpy as np
from numpy import linalg as LA
import matplotlib
import time
from PIL import Image

"""

im = Image.fromarray(arr, mode="L")
im.show()
"""

def show_img(img):
    # shows image as png
    # img must be a 2576 1d vector
    img.resize((46,56))
    im = Image.fromarray(np.transpose(img), mode="L")
    im.show()


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

# compute the covariance matrix for low-dimensional computation
start = time.time()
low_cov_mat = np.cov(training_data)
end = time.time()
print("time required to compute low-dim-cov-mat : ",end - start)
print("length of covariance matrix", len(low_cov_mat))




# compute the eigenvalues and eigenvectors of the covariance matrix
# w -> eigenvalues
# v -> eigenvectors
start = time.time()
low_w, low_v = LA.eig(low_cov_mat)

# reconstruct original eignevectors
low_computed_u = training_data.T.dot(low_v) # is probably incredibly wrong
end = time.time()
print("time required to compute eigenvalues and eigenvectors of low-dim-cov-mat : ",end-start)

# create matrix of eigen vectors with nonzero eigen values
nz_eigenvectors = low_computed_u.T[low_w != 0]
print("length of non-zero eigenvalue array: ", len(nz_eigenvectors))
# print(nz_eigenvectors)

# sort form largest to smallest eigen vector
low_eigenval_vec = [(x,y) for x, y in zip(low_w, low_computed_u)]
sorted_low = sorted(low_eigenval_vec, key=lambda eigenv: eigenv[0], reverse=True)
# print(sorted_low)

# compute the covariance matrix for normal PCA computation
normal_cov_mat = np.cov(np.transpose(training_data)) # need timing
print("Length of normal cov matrix", len(normal_cov_mat))

# compute eigenvalues and vectors for normal PCA computation
# normal_t -> eigenvalues
# normal_u -> eigenvectors
normal_t, normal_u = LA.eig(normal_cov_mat) # need to do timing

# get the eigen vectors corresponding to the largest 416 eigen values
normal_eigenval_vec = [(x,y) for x, y in zip(normal_t, normal_u)]
sorted_normal = sorted(normal_eigenval_vec, key=lambda eigenv: eigenv[0], reverse=True)
sorted_normal = sorted_normal[0:416]

# check that the eigen values and the eigen vectors are the same from both computations
print("are the computed eigen values and vectors the same?")
print("size of sorted", len)
if np.array_equal(sorted_normal, sorted_low):
	print("yoyoyo")
else:
	print("f off")






