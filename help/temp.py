import prepareDataset
import numpy as np
import torch
import time
import torch.nn as nn
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error


def myPCA(X, reducedDimension):
    startTime = time.time()
    X = X.numpy()
    X = X.T
    print("PCA Running, Data format should have shape dimension*numSamples")
    # X = np.random.random((originalDimension,numSamples))
    originalDimension, numSamples = X.shape
    # X = normalize(X, axis=0, norm ='l2')
    meanVector = [np.mean(X[i,:]) for i in range(originalDimension)]
    scatterMatrix = np.zeros((originalDimension,originalDimension))
    for i in range(X.shape[1]):
        scatterMatrix += (X[:,i].reshape(originalDimension,1) - meanVector).dot((X[:,i].reshape(originalDimension,1) - meanVector).T)
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatterMatrix)
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    lis = []
    count = 0
    sumRD = 0
    sumAll = 0

    for i in eig_pairs:
        count = count + 1
        if count <= reducedDimension:
            lis.append(i[1].reshape(originalDimension,1))
            sumRD += i[0]
        sumAll += i[0]
    matrix_w = np.hstack(lis)
    Y = matrix_w.T.dot(X)
    newX = np.linalg.pinv(matrix_w.T).dot(Y)
    MSE = mean_squared_error(X, newX)
    print("MSE = {}".format(MSE))
    endTime = time.time()
    print("Time taken by PCA",(endTime-startTime))
    fp = sumRD / sumAll * 100
    print("FPOpppp", fp)
    return torch.from_numpy(Y.T)

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
torch.manual_seed(7)


# anchorFrames = prepareDataset.getPersonFrames(seqRootRGB, personIdxDict[1], 1, 25)
# print anchorFrames
# for i in range(1,11):
    # print myPCA(anchorFrames, 10)
def vec():
    tic = time.time()
    alpha = torch.FloatTensor([0.4])
    torch.manual_seed(7)

    if torch.cuda.is_available():
        alpha = alpha.cuda()
    alpha = Variable(alpha)
    a = Variable(torch.randn(1, 16, 128))
    print a[0][0]
    b = Variable(torch.randn(1, 16, 128))
    c = Variable(torch.randn(1, 16, 128))

    loss = torch.zeros(1)
    if torch.cuda.is_available():
        loss = loss.cuda()
    loss = Variable(loss)
    zero = torch.zeros(1)
    if torch.cuda.is_available():
        zero = zero.cuda()
    zero = Variable(zero)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    z = torch.sum(torch.max(zero, alpha - cos(a[0], b[0]) + cos(a[0], c[0])))
    # print x
    # print y
    print z
    toc = time.time()
    print("Vector time: " + str((toc-tic)*1000) + "ms")

def lo():
    tic = time.time()
    alpha = torch.FloatTensor([0.4])
    torch.manual_seed(7)

    if torch.cuda.is_available():
        alpha = alpha.cuda()
    alpha = Variable(alpha)
    a = Variable(torch.randn(1, 16, 1, 128))
    print a[0][0]
    b = Variable(torch.randn(1, 16, 1, 128))
    c = Variable(torch.randn(1, 16, 1, 128))

    loss = torch.zeros(1)
    if torch.cuda.is_available():
        loss = loss.cuda()
    loss = Variable(loss)
    zero = torch.zeros(1)
    if torch.cuda.is_available():
        zero = zero.cuda()
    zero = Variable(zero)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    for j in range(16):
        zero = torch.zeros(1)
        if torch.cuda.is_available():
            zero = zero.cuda()
        zero = Variable(zero)
        loss = loss + torch.max(zero, alpha - cos(a[0][j], b[0][j]) + cos(a[0][j], c[0][j]))
    print loss
    toc = time.time()
    print("Lo time: " + str((toc-tic)*1000) + "ms")

def getOrderStats():
    v1 = torch.randn(1, 128)
    v2 = torch.randn(1, 128)

    for i in range(1, 8):
        x = torch.randn(1, 128)
        y = torch.randn(1, 128)
        v1 = torch.cat((v1, x), dim=0)
        v2 = torch.cat((v2, y), dim=0)
    print v1

getOrderStats()
