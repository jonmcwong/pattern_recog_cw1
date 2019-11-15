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
    if len(input[0]) == 2576 and len(mean_face[0]) == 2576 and len(eig[0]) == 2576:
        if n < len(eig[0]):
            delta = input - mean_face
            eig_face = eig[:n,:]
#             print("delta ",delta.shape)
#             print("eigface ",eig_face.shape)
            weights = np.matmul(delta, eig_face.T)
#             print("weights ",weights.shape)
#             print("eigface ",eig_face.shape)
            reconstructed_faces = np.matmul(weights, eig_face)
            face = np.empty(input.shape)
            for i in range(len(reconstructed_faces)):
                face[i][:] = reconstructed_faces[i][:]+mean_face

            return face
    else:
        raise ValueError("Testing data / Mean dimension error.")

def plot_err(training_data, testing_data, mean_face, eig, image):
    if len(training_data[0]) == 2576 and len(training_data[0]) == 2576 and len(training_data[0]) == 2576 and len(training_data[0]) == 2576:
        
        index = []
        for i in range(1,len(eig)):
            faces = reconstruct(testing_data, mean_face, eig, i)
            temp = []
            for j in range(len(testing_data)):
                temp.append(get_err(testing_data[j], faces[j]))
            index.append(sum(temp)/len(temp))
        plt.plot(index)
        plt.title(("Average absolute error of testing data["+str(image)+"]"))
        plt.xlabel("Number of eigenvectors used.")
        plt.ylabel("Average absolute error of testing data") 
    else:
        raise ValueError("Input Dimension Error")

def eigen_analysis(w1,v1,w2,v2):
    plt.subplot(311)
    plt.plot(w1[:415])
    plt.subplot(312)
    plt.plot(w2[:415])
    plt.subplot(313)
    plt.plot(w1[:415]-w2[:415])
    
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
    
def class_rate(training_data, reconstructed):
    if training_data[0].shape == reconstructed[0].shape:
        dist, indx = nn_class(training_data, reconstructed)
        result = []
        for i in range(len(indx)):
            if int(i/2)*8 <= indx[i] <= (int(i/2)+1)*8:
                result.append(1)
            else:
                result.append(0)
        return result
    else:
        raise ValueError("training_data / reconstructed dimension error")

def plot_class_rate(training_data, testing_data, mean_face, eig):
    if len(training_data[0]) == 2576 and len(testing_data[0]) == 2576 and len(mean_face[0]) == 2576 and len(eig[0]) ==2576:
        Y = []
        for i in range(1,len(eig)):
            faces = reconstruct(testing_data, mean_face, eig, i)
            reconstructed_weights = np.matmul((testing_data), eig[:i,:].T)
            training_weights = np.matmul((training_data), eig.T)
            result = class_rate(training_weights[:,:i], reconstructed_weights)
            Y.append(100*np.sum(result)/len(result))

        plt.plot(Y)
        plt.title(("Classification rate of testing data"))
        plt.xlabel("Number of eigenvectors used")
        plt.ylabel("Classification Rate (%)")
    else:
       raise ValueError("training / testing / mean dimension error")
   
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
    
def combined_mean(ds0,ds1):
    _,n0 = ds0.shape
    _,n1 = ds1.shape    
    mu0 = ds0.mean(axis=0)
    mu1 = ds1.mean(axis=0)
    mu3 = (n0*mu0 + n1*mu1)/(n0+n1)
    return mu3

def combined_cov(ds0,ds1):
    _,n0 = ds0.shape
    _,n1 = ds1.shape    
    mu0 = ds0.mean(axis=0)
    mu1 = ds1.mean(axis=0)
    s0 = (1/n0)*np.dot(ds0.T,ds0)
    s1 = (1/n1)*np.dot(ds1.T,ds1)   
    combined_cov =  (n0/(n0+n1))*s0 + (n1/(n0+n1))*s1 + ((n0*n1)/(n0+n1)**2)*np.dot(mu0-mu1,(mu0-mu1).T)
    return combined_cov

def combined_ds(ds0,ds1):
    n0,d0 = ds0.shape
    n1,d1 = ds1.shape
    print("d0 : ",d0)
    print("n0 : ",n0)
    combined = np.empty([n0+n1, d0])
    print("combined shape : ",combined.shape)

    #for each class:
    for d_class in range(52):
        #get nth 10% from d0
        f0 = n0/52
        f1 = n1/52
        print(f0)
        print(f1)
        start0= int(d_class*(f0+f1))
        end0 = int(d_class*(f0+f1)+f0)
        start1 = int(d_class*(f0+f1)+f0)
        end1 = int((d_class+1)*(f0+f1))
        print("start0", start0)
        print("end0", end0)
        print("start1", start1)
        print("end1", end1)
        
        combined[start0:end0] = ds0[int(d_class*f0):int((d_class+1)*f0)]
        combined[start1:end1] = ds1[int(d_class*f1):int((d_class+1)*f1)]
        #get nth 10% from d1
    
    
#     return np.concatenate((ds0,ds1), axis=0)
    return combined

def merge_dataset(ds0, ds1):
    #needs a different name
    new_mu = combined_mean(ds0, ds1)
    new_cov = combined_cov(ds0,ds1)
    combined_training = combined_ds(ds0,ds1)
    mu0 = ds0.mean(axis=0)
    mu1 = ds1.mean(axis=0)
    _,n0 = ds0.shape
    _,n1 = ds1.shape
    
    s0 = (1/n0)*np.dot(ds0,ds0.T)
    w0, v0 = LA.eig(s0)
    p0 = np.dot(ds0.T,v0)
    p0 /= LA.norm(p0,ord=2,axis=0)
    id = np.argsort(np.abs(w0))[::-1]
    w0 = w0[id]
    p0 = p0[:,id].real
    
    s1 = (1/n0)*np.dot(ds1,ds1.T)
    w1, v1 = LA.eig(s1)
    p1 = np.dot(ds1.T,v1)
    p1 /= LA.norm(p1,ord=2,axis=0)
    id = np.argsort(np.abs(w1))[::-1]
    w1 = w1[id]
    p1 = p1[:,id].real

    phi,_r = LA.qr(np.concatenate((p0,p1,(mu0-mu1).reshape(-1,1)), axis = 1))
    
    phi /= LA.norm(phi,ord=2,axis=0)
    
    temp = np.matmul(np.matmul(phi.T,new_cov),phi)
    
    delta,r = LA.eig(temp)
    
    r /= LA.norm(r,ord=2,axis=0)
    id = np.argsort(np.abs(delta))[::-1]
    delta = delta[id]
    r = r[:,id].real
    
    p3 = np.matmul(phi,r)
    
    
    return combined_training, p3.T, new_mu.T.reshape(1,-1), new_cov
    
    

