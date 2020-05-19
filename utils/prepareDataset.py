import buildModel

import os
import time
import torch
import timeit
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error

datasetNo = 1

def getSequenceImageFiles(seqRoot):
    imgsList = sorted(os.listdir(seqRoot))
    if imgsList[0] == '.DS_Store':
        imgsList.pop(0)
    return imgsList

def getPersonDirsList(seqRootDir):
    if datasetNo == 1:
        firstCameraDirName = 'cam1'
    else:
        firstCameraDirName = 'cam_a'
    seqRootDir = os.path.join(seqRootDir,firstCameraDirName)
    dirsList = sorted(os.listdir(seqRootDir))
    if dirsList[0] == '.DS_Store':
        dirsList.pop(0)
    return dirsList

def transformDataset(dataset):
    tempPerson = []
    for i in range(300):
        tempCam = []
        tempCam.append(dataset[0][i])
        tempCam.append(dataset[1][i])
        tempPerson.append(tempCam)
    return tempPerson

def myPCA(X, reducedDimension):
    # startTime = time.time()
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
    for i in eig_pairs:
        lis.append(i[1].reshape(originalDimension,1))
        count = count + 1
        if count == reducedDimension:
            break
    matrix_w = np.hstack(lis)
    Y = matrix_w.T.dot(X)
    newX = np.linalg.pinv(matrix_w.T).dot(Y)
    MSE = mean_squared_error(X, newX)
    # print("MSE = {}".format(MSE))
    # endTime = time.time()
    # print("Time taken by PCA",(endTime-startTime))
    return torch.from_numpy(Y.T)

def generateTriplets(nTotalPersons, testTrainSplit):
    splitPoint = int(nTotalPersons * testTrainSplit)
    inds = torch.randperm(nTotalPersons)
    inds += 1
    trainInds = inds[0:splitPoint]
    # print trainInds
    # testInds = inds[(splitPoint):nTotalPersons+1]
    # trainInds = torch.load("/Users/prateek/8thSem/features/trainInds.pt")
    # testInds = torch.load("/Users/prateek/8thSem/features/testInds.pt")
    # random.shuffle(trainInds)
    # random.shuffle(testInds)
    trainTriplets = []
    testTriplets = []
    for person in trainInds:
        triplet = [0, 0, 0]
        triplet[0] = person
        triplet[1] = person
        while(1):
            triplet[2] = random.choice(trainInds)
            if triplet[2] == person:
                triplet[2] = random.choice(trainInds)
            else:
                break
        trainTriplets.append(triplet)
    # for person in testInds:
    #     triplet = [0, 0, 0]
    #     triplet[0] = person
    #     triplet[1] = person
    #     while(1):
    #         triplet[2] = random.choice(testInds)
    #         if triplet[2] == person:
    #             triplet[2] = random.choice(testInds)
    #         else:
    #             break
    #     testTriplets.append(triplet)
    return trainTriplets, testTriplets

def loadImage(filename):
    print filename
    img = Image.open(filename)
    img = img.convert('YCbCr')
    imgY, imgU, imgV = img.split()
    imgY = transforms.ToTensor()(imgY)
    imgU = transforms.ToTensor()(imgU)
    imgV = transforms.ToTensor()(imgV)
    meanY = torch.mean(imgY)
    meanU = torch.mean(imgU)
    meanV = torch.mean(imgV)
    stdY = torch.std(imgY)
    stdU = torch.std(imgU)
    stdV = torch.std(imgV)
    imgY = (imgY - meanY) / stdY
    imgU = (imgU - meanU) / stdU
    imgV = (imgV - meanV) / stdV
    imgY = transforms.ToPILImage()(imgY)
    imgU = transforms.ToPILImage()(imgU)
    imgV = transforms.ToPILImage()(imgV)
    img = transforms.Pad((80, 48), fill=0)(img)
    img = transforms.ToTensor()(img)
    return img
    # img = img.unsqueeze_(0)
    # if torch.cuda.is_available():
    #     img = img.cuda()
    # img = Variable(img)
    # return buildModel.cnn(img)

def loadSequenceImages(cameraDir,filesList, actualFrameCount):
    print("Files to be loaded", cameraDir, filesList[0:actualFrameCount])
    print("Total frames to be loaded = {}".format(actualFrameCount))
    # nImgs = len(filesList)
    imgList = torch.Tensor(actualFrameCount, 1, 4096)
    for (i, file) in enumerate(filesList):
        if i == actualFrameCount:
            break
        startTime = time.time()
        filename = os.path.join(cameraDir,file)
        imgList[i] = loadImage(filename).data
        endTime = time.time()
        print("Time Taken to load one image",(endTime-startTime))
    imgList = torch.cat(imgList, dim=0)
    return imgList

def getPersonFrames(datasetRootDir, personDir, cam, actualFrameCount):
    letter = ['a','b']
    if datasetNo == 1:
        cameraDirName = 'cam' + str(cam)
    else:
        cameraDirName = 'cam_' + letter[cam-1]
    seqRoot = os.path.join(datasetRootDir,cameraDirName,personDir)

    # print("Print seqRoot", seqRoot)
    seqImgs = getSequenceImageFiles(seqRoot)
    personFramesData = loadSequenceImages(seqRoot, seqImgs, actualFrameCount)
     # personFramesData = myPCA(personFramesData, 1024)
    personFramesData = myPCA(personFramesData, 64)
    return personFramesData.unsqueeze_(0)

def prepareDS(datasetRootDir):
    personFramesDict = {}
    personIdxDict = {}
    dataset = [[], []]
    personDirs = getPersonDirsList(datasetRootDir)
    # print("Print personDir", personDirs)
    nPersons = len(personDirs)
    # print nPersons
    letter = ['a','b']
    for (i, pdir) in enumerate(personDirs):
        # print("Print Person No:", i, pdir)
        personIdxDict[i+1] = pdir
        personFramesDict[pdir] = [0, 0]
        for cam in range(1, 3):
            if datasetNo == 1:
                cameraDirName = 'cam' + str(cam)
            else:
                cameraDirName = 'cam_' + letter[cam-1]
            seqRoot = os.path.join(datasetRootDir,cameraDirName,pdir)
            # print("Print seqRoot", seqRoot)
            seqImgs = getSequenceImageFiles(seqRoot)
            # print("Print seqImgs", seqImgs)
            personFramesDict[pdir][cam-1] = len(seqImgs)
            # dataset[cam-1].append(loadSequenceImages(seqRoot,seqImgs))
    # dataset = transformDataset(dataset)
    # dataset = reduceDataset(dataset)
    return personIdxDict, personFramesDict
