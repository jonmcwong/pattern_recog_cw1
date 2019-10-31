import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib
import time
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

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

# get mean image matrix
mean_face = raw_data.mean(axis=1).reshape(-1,1)

A = raw_data - mean_face
S = (1/N)*np.dot(np.transpose(A),A)

w, v = LA.eig(S)

u = np.dot(A,v)

id = np.argsort(np.abs(w))[::-1]

w = w[id]
u = np.real(u[:,id])



for i in range(10):
    asdf = u[:,i].reshape(46,56)
    im = Image.fromarray(asdf)
    im.show()

# for new image newdat
delta = newdat - mean_face
weights = np.dot(np.transpose(u), delta)


#reconstruct
img = np.dot(u, weights)
