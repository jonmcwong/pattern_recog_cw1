import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib
import time
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sympy


def show_img(img):
    # shows image as png
    # img must be a 2576 1d vector
    temp = img[:]
    temp.reshape((46,56))
    im = Image.fromarray(temp)
    im.show()

# load data
mat = scipy.io.loadmat('face.mat')
raw_data = mat['X']

D,N = raw_data.shape

raw_data = np.transpose(raw_data)

training_data = np.empty([int(520*0.8), 2576])
testing_data = np.empty([int(520*0.2), 2576])

# create training and test data
for x in range(52):
        # 8/2 ratio for training and testing datasets
	training_data[x*8:(x+1)*8] = raw_data[x*10:x*10+8]
	testing_data[x*2:(x+1)*2] = raw_data[x*10+8:(x+1)*10]

raw_data = np.transpose(raw_data)
training_data = np.transpose(training_data)
testing_data = np.transpose(testing_data)

# get mean image matrix
mean_face = training_data.mean(axis=1).reshape(-1,1)
asdf = mean_face.reshape(46,56)
im = Image.fromarray(asdf)
im.show()




A = training_data - mean_face
#S = (1/N)*np.dot(np.transpose(A),A)
S = np.cov(np.transpose(A))

w, v = LA.eig(S)

v = v/v.max(axis=0)

# u = principal components
u = np.dot(A,v)
u = u/u.max(axis=0)

id = np.argsort(np.abs(w))[::-1]
w = w[id]
u = np.real(u[:,id])
u_norm = u
"""
for i in range(10):
    asdf = u_norm[:,i].reshape(46,56)
    im = Image.fromarray(asdf)
    im.show()
"""

input = testing_data

# for new image testing_data
# center the testing data
delta = input - mean_face

pc_score = np.empty([len(delta[0]),len(u_norm[0])])


for i in range(len(delta[0])):
    for j in range(len(u_norm[0])):
        pc_score[i,j] = np.dot(u_norm[:,j], delta[:,i])


print("pc score: ", pc_score.shape)
print("u_norm: ", u_norm.shape)

img = np.empty([len(pc_score[0]),len(u_norm[0])])
print(img.shape)

for j in range(len(u_norm[0])):
    for i in range(len(pc_score[0])):
        print(pc_score[j][i])
        print(u_norm[:,j])
        img[:,j] += pc_score[j][i]*u_norm[:,j]


print(img.shape)

exit()
# reconstruction




print("reconstructed : ", img.shape)
for it in range(1):
    print(img[:,it])
    asdf = img[:,it].reshape(46,56)
    ima = Image.fromarray(asdf)
    ima.show()
