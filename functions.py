import scipy.io
import numpy as np
from numpy import linalg as LA
import matplotlib
import time
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def show_img(img):
    temp = img.copy()
    temp.resize((46,56))
    im = Image.fromarray(temp.T)
    im.show()

def lowdim_pca(train, mean_face): 
    A = train - mean_face
    D,N = A.shape
    start = time.time()
    S = (1/N)*np.dot(A.T,A)
    print("shape of S : ",S.shape)
#     print("rank of lowdim cov: ", LA.matrix_rank(S))
    w, v = LA.eig(S)
    v /= LA.norm(v,ord=2,axis=0)
    print("shape of w", w.shape)
    print("shape of v", v.shape)
    # u = principal components
    u = np.dot(A,v)
    u /= LA.norm(u,ord=2,axis=0)

    id = np.argsort(np.abs(w))[::-1]
    w = w[id]
    u = u[:,id].real
    end = time.time()
    print("low dimension pca took ", end-start ," seconds.")
    # return eigen vectors sorted from largest to smallest
    return w, u

def normal_pca(train, mean_face):
    A = train - mean_face
    D,N = A.shape
    start = time.time()
    S = (1/N)*np.dot(A,A.T)

#     print("rank of S: ", LA.matrix_rank(S))
#     print("S is symmetric: ", S == S.T)
#     print()
#     print("S is real: ", S.imag == 0)
    
    w,v = LA.eig(S)
    v /= LA.norm(v,ord=2,axis=0)
    # nz_u = principal components with non-zero eigenvals
#     print("number of zero eigen vals: ", np.sum(w != 0))
    nz_u = v[w != 0]
    nz_u /= LA.norm(nz_u, ord=2, axis=0)
    nz_w = w[w != 0]
#     print("eigenvalues: ", nz_w)
#     print("complex eigen vals are: ", nz_w[nz_w.imag != 0])
    id = np.argsort(np.abs(nz_w))[::-1]
    nz_w = nz_w[id].real
    nz_u = nz_u[:,id].real
    end = time.time()
    print("normal pca took ", end-start ," seconds.")
    # return non-zero eigen vectors sorted from largest to smallest
    return nz_w, nz_u    


def get_err(x, y):
    # they must be the same dimension
    err = np.absolute(x - y)
    return np.sum(err)/len(x)

def reconstruct(input, mean_face, eig, n):
    if n < len(eig[0]):
        delta = input - mean_face
        eig_face = eig[:,:n]
        weights = np.matmul(delta.T, eig_face)
        reconstructed_faces = np.matmul(weights, eig_face.T)
        face = np.empty(input.T.shape)
        for i in range(len(reconstructed_faces)):
            face[i][:] = reconstructed_faces[i][:]+mean_face.T

        return face

def nn_class(X, Y):
    # X = training data
    # Y = reconstructed testing data
    # this function returns the nearest data point of Y in X
    # X and Y must have the same shape
    # indices denotes the index of the nearest neighbour of Y in X
    # indices denotes the corresponding distance 
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(Y)
    return distances, indices

def plot_err(training_data, testing_data, mean_face, eig, image):
    index = []
    for i in range(1,len(eig[0])):
        faces = reconstruct(testing_data, mean_face, eig, i)
        temp = []
        for j in range(len(testing_data[1,:])):
            temp.append(get_err(testing_data[:,j], faces[j]))
        index.append(sum(temp)/len(temp))
    plt.plot(index)
    plt.xlabel("Number of eigenvectors used.")
    plt.ylabel("Average absolute error of testing data")

def eigen_analysis(w1,v1,w2,v2):
    plt.subplot(311)
    plt.plot(w1[:415])
    plt.subplot(312)
    plt.plot(w2[:415])
    plt.subplot(313)
    plt.plot(w1[:415]-w2[:415])
    
def class_rate(training_data, reconstructed):
    print("training_data.T: ", training_data.T.shape)
    print("reconstructed: ", reconstructed.shape)
    dist, indx = nn_class(training_data.T, reconstructed)
    result = []
    for i in range(len(indx)):
        if int(i/2)*8 <= indx[i] <= (int(i/2)+1)*8:
            result.append(True)
        else:
            result.append(False)
    return result

def plot_class_rate(training_data, testing_data, mean_face, eig, image):
    Y = []
    for i in range(1,len(eig[0])):
        faces = reconstruct(testing_data, mean_face, eig, i)
        result = class_rate(training_data, faces)
        Y.append(100*np.sum(result)/len(result))
    plt.plot(Y)
    plt.xlabel("Number of eigenvectors used")
    plt.ylabel("Classification Rate (%)")
   
# get classifications accuracies for varying Mlda
def vary_Mlda(training_weights,testing_weights):
    acc = []
    for i in range(1, len(testing_weights[0]) + 1):
        result = class_rate(training_weights[:i].real, testing_weights[:i].real)
        acc.append(100*np.sum(result)/len(result))
    return acc

# plot data
def plot_data(title, xlabel, ylabel, y_data):
    plt.plot(y_data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

        
    
