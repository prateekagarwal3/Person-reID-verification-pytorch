import prepareDataset

import os
import copy
import random
from itertools import count
from collections import namedtuple

import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

frameDropThreshold = 0.625

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())

def getPairInfo(pair):
    maxFrameCount = 192
    id1 = pair[0][0]
    cam1 = pair[0][1]
    id2 = pair[1][0]
    cam2 = pair[1][1]

    pid = torch.IntTensor([id1, id2])
    fc1 = personFramesDict[personIdxDict[id1]][cam1]
    fc2 = personFramesDict[personIdxDict[id2]][cam2]
    fcMax = max(fc1, fc2)
    framesCount = torch.IntTensor([fc1, fc2])
    threshold = torch.IntTensor([int(fc1*frameDropThreshold), int(fc2*frameDropThreshold)])
    initialState = torch.ones(2, maxFrameCount)
    if fc1 < maxFrameCount:
        initialState[0, fc1:maxFrameCount] = 0
    if fc2 < maxFrameCount:
        initialState[1, fc2:maxFrameCount] = 0

    tempDict = {}
    tempDict['A'] = [i for i in range(0, fc1-19)]
    tempDict['B'] = [i for i in range(0, fc2-19)]
    for i in range(0, len(tempDict['A'])-4, 5):
        tempDict['A'][i+1] = -1
        tempDict['A'][i+2] = -1
        tempDict['A'][i+3] = -1
        tempDict['A'][i+4] = -1
    for i in range(0, len(tempDict['A'])):
        if i % 5 != 0 and tempDict['A'][i] != -1:
            tempDict['A'][i] = -1
    for i in range(0, len(tempDict['B'])-4, 5):
        tempDict['B'][i+1] = -1
        tempDict['B'][i+2] = -1
        tempDict['B'][i+3] = -1
        tempDict['B'][i+4] = -1
    for i in range(0, len(tempDict['B'])):
        if i % 5 != 0 and tempDict['B'][i] != -1:
            tempDict['B'][i] = -1
    framesDropInfo = torch.IntTensor(2, maxFrameCount)
    framesDropInfo[0, 0:fc1-19] = torch.IntTensor(tempDict['A'])
    framesDropInfo[0, fc1-19:maxFrameCount] = -1
    framesDropInfo[1, 0:fc2-19] = torch.IntTensor(tempDict['B'])
    framesDropInfo[1, fc2-19:maxFrameCount] = -1

    if torch.cuda.is_available():
        pid = pid.cuda()
        framesDropInfo = framesDropInfo.cuda()
        framesCount = framesCount.cuda()
        threshold = threshold.cuda()
        initialState = initialState.cuda()
    return pid, framesDropInfo, framesCount, threshold, initialState

def generatePairs(testTriplets):
    # print testTriplets
    testPairs = []
    for i in range(testTriplets.size(0)):
        tempPair = [[testTriplets[i][0], 0], [testTriplets[i][1], 1]]
        testPairs.append(tempPair)
        tempPair = [[testTriplets[i][1], 1], [testTriplets[i][2], 1]]
        testPairs.append(tempPair)
        tempPair = [[testTriplets[i][0], 0], [testTriplets[i][2], 1]]
        testPairs.append(tempPair)
    return testPairs

def countOnes(state):
    fc1 = 0
    fc2 = 0
    for i in range(state.size(1)):
        if int(state[0][i]) is 1:
            fc1 += 1
    for i in range(state.size(1)):
        if int(state[1][i]) is 1:
            fc2 += 1
    return fc1, fc2

def getState(pid, framesCount, state, framesDropInfo):
    fc1, fc2= countOnes(state)
    if fc1 < sequence_length or fc2 < sequence_length
        return torch.randn(2, 128).cuda() if torch.cuda.is_available() else torch.randn(2, 128)
    anchorRGBFeatures = torch.load(dirPath + 'features/featuresRGB/cam1/' + str(pid[0].data[0]) + '.pt')
    positiveRGBFeatures = torch.load(dirPath + 'features/featuresRGB/cam2/' + str(pid[1].data[0]) + '.pt')
    negativeRGBFeatures = torch.load(dirPath + 'features/featuresRGB/cam2/' + str(pid[2].data[0]) + '.pt')
    anchorFrames = torch.Tensor(fc1, 64)
    positiveFrames = torch.Tensor(fc2, 64)
    negativeFrames = torch.Tensor(fc3, 64)

    anchorOPFeatures = torch.load(dirPath + 'features/featuresOP/cam1/' + str(pid[0].data[0]) + '.pt')
    positiveOPFeatures = torch.load(dirPath + 'features/featuresOP/cam2/' + str(pid[1].data[0]) + '.pt')
    negativeOPFeatures = torch.load(dirPath + 'features/featuresOP/cam2/' + str(pid[2].data[0]) + '.pt')
    anchorOPFrames = torch.Tensor(fc1, 64)
    positiveOPFrames = torch.Tensor(fc2, 64)
    negativeOPFrames = torch.Tensor(fc3, 64)
    k = 0
    for i in range(state.size(1)):
        if int(state[0][i]) == 1:
            anchorFrames[k] = anchorRGBFeatures[i]
            anchorOPFrames[k] = anchorOPFeatures[i]
            k +=1
    k = 0
    for i in range(state.size(1)):
        if int(state[1][i]) == 1:
            positiveFrames[k] = positiveRGBFeatures[i]
            positiveOPFrames[k] = positiveOPFeatures[i]
            k +=1

    k = 0
    for i in range(state.size(1)):
        if int(state[2][i]) == 1:
            negativeFrames[k] =  negativeRGBFeatures[i]
            negativeOPFrames[k] =  negativeOPFeatures[i]
            k +=1

    anchorFC = anchorFrames.size(0)
    positiveFC = positiveFrames.size(0)
    negativeFC = negativeFrames.size(0)
    maxFC = max(anchorFC, positiveFC, negativeFC)
    anchorBatchSize = anchorFC / sequence_length + 1
    positiveBatchSize = positiveFC / sequence_length + 1
    negativeBatchSize = negativeFC / sequence_length + 1
    maxBatchSize = max(anchorBatchSize, positiveBatchSize, negativeBatchSize)
    anchorIP = torch.Tensor(maxBatchSize, sequence_length, 64)
    positiveIP = torch.Tensor(maxBatchSize, sequence_length, 64)
    negativeIP = torch.Tensor(maxBatchSize, sequence_length, 64)
    for i in range(maxBatchSize):
        if i < anchorBatchSize-1:
            anchorIP[i] = anchorFrames[sequence_length*i : sequence_length*(i+1)]
        else:
            anchorIP[i] = anchorFrames[0 : sequence_length]
    for i in range(maxBatchSize):
        if i < positiveBatchSize-1:
            positiveIP[i] = positiveFrames[sequence_length*i : sequence_length*(i+1)]
        else:
            positiveIP[i] = positiveFrames[0 : sequence_length]
    for i in range(maxBatchSize):
        if i < negativeBatchSize-1:
            negativeIP[i] = negativeFrames[sequence_length*i : sequence_length*(i+1)]
        else:
            negativeIP[i] = negativeFrames[0 : sequence_length]
    H, Hp, Hn = tripletRNNRGB(anchorIP, positiveIP, negativeIP)
    H = torch.cat(H.data, dim=0)
    Hp = torch.cat(Hp.data, dim=0)
    Hn = torch.cat(Hn.data, dim=0)
    anchorRGBFeatures = H[0:anchorFC]
    positiveRGBFeatures = Hp[0:positiveFC]
    negativeRGBFeatures = Hn[0:negativeFC]

    anchorOPFC = anchorOPFrames.size(0)
    positiveOPFC = positiveOPFrames.size(0)
    negativeOPFC = negativeOPFrames.size(0)
    maxOPFC = max(anchorOPFC, positiveOPFC, negativeOPFC)
    anchorOPBatchSize = anchorOPFC / sequence_length + 1
    positiveOPBatchSize =  positiveOPFC / sequence_length + 1
    negativeOPBatchSize = negativeOPFC / sequence_length + 1
    maxOPBatchSize = max(anchorOPBatchSize, positiveOPBatchSize, negativeOPBatchSize)
    anchorOPIP = torch.Tensor(maxOPBatchSize, sequence_length, 64)
    positiveOPIP = torch.Tensor(maxOPBatchSize, sequence_length, 64)
    negativeOPIP = torch.Tensor(maxOPBatchSize, sequence_length, 64)
    for i in range(maxBatchSize):
        if i < anchorOPBatchSize-1:
            anchorOPIP[i] = anchorOPFrames[sequence_length*i : sequence_length*(i+1)]
        else:
            anchorOPIP[i] = anchorOPFrames[0 : sequence_length]
    for i in range(maxOPBatchSize):
        if i < positiveOPBatchSize-1:
            positiveOPIP[i] = positiveOPFrames[sequence_length*i : sequence_length*(i+1)]
        else:
            positiveOPIP[i] = positiveOPFrames[0 : sequence_length]
    for i in range(maxOPBatchSize):
        if i < negativeOPBatchSize-1:
            negativeOPIP[i] = negativeOPFrames[sequence_length*i : sequence_length*(i+1)]
        else:
            negativeOPIP[i] = negativeOPFrames[0 : sequence_length]

    HOP, HpOP, HnOP = tripletRNNOP(anchorOPIP, positiveOPIP, negativeOPIP)
    HOP = torch.cat(HOP.data, dim=0)
    HpOP = torch.cat(HpOP.data, dim=0)
    HnOP = torch.cat(HnOP.data, dim=0)
    anchorOPFeatures = HOP[0:anchorFC]
    positiveOPFeatures = HpOP[0:positiveFC]
    negativeOPFeatures = HnOP[0:negativeFC]
    anchorFeatures = torch.mean(torch.cat((anchorRGBFeatures, anchorOPFeatures), dim=1), dim=0).view(-1, 128)
    positiveFeatures = torch.mean(torch.cat((positiveRGBFeatures, positiveOPFeatures), dim=1), dim=0).view(-1, 128)
    negativeFeatures = torch.mean(torch.cat((negativeRGBFeatures, negativeOPFeatures), dim=1), dim=0).view(-1, 128)

    features = torch.cat((anchorFeatures, positiveFeatures, negativeFeatures), dim=0)

    if torch.cuda.is_available():
        features = features.cuda()

    return features

def getAction(pair, pid, framesCount, state, framesDropInfo, model):

    stateReduced = getState(pid, framesCount, state, framesDropInfo)
    qValues = model(stateReduced).data
    maxValue = qValues[0][0]
    maxIndex = torch.IntTensor([0, 0])
    # print("printing qvalues size", qValues.size())
    for i in range(qValues.size(0)):
        for j in range(qValues.size(1)):
            if j * 5 < int(framesCount[i]) and int(framesDropInfo[i][j*5]) != -1:
                if maxValue < qValues[i][j]:
                    # print("loop running")
                    maxValue = qValues[i][j]
                    maxIndex = torch.IntTensor([i, j*5])
    return maxIndex

def performAction(state, action, threshold, pid, framesDropInfo, framesCount):
    nextState = state.clone()
    for i in range(0, 10, 2):
        nextState[action[0], (i + action[1])] = 0
    for i in range(0, 10, 2):
        if action[1]+i in framesDropInfo[action[0]]:
            framesDropInfo[action[0], (action[1]+i)] = -1
    for i in range(1, 10, 2):
        if action[1] - i in framesDropInfo[action[0]]:
            framesDropInfo[action[0], (action[1]-i)] = -1

    reward = findReward(state, nextState, pid, framesCount)
    if torch.cuda.is_available():
        nextState = nextState.cuda()
    return nextState, framesDropInfo, reward

def getStateValues(pid, framesCount, state, framesDropInfo, model):
    stateReduced = getState(pid, framesCount, state, framesDropInfo)
    qValues = model(stateReduced).data
    maxValue = qValues[0][0]
    maxIndex = torch.IntTensor([0, 0])
    for i in range(qValues.size(0)):
        for j in range(qValues.size(1)):
            if j * 5 < int(framesCount[i]) and int(framesDropInfo[i][j*5]) != -1:
                if maxValue < qValues[i][j]:
                    # print("loop running")
                    maxValue = qValues[i][j]
                    # print maxValue
                    # print qValues[i][j]
                    maxIndex = torch.IntTensor([i, j*5])
    if torch.cuda.is_available():
        maxValue = maxValue.cuda()
    return Variable(maxValue)

def checkTerminal(pid, framesCount, state, framesDropInfo, model):
    done = 0
    if getStateValues(pid, framesCount, state, framesDropInfo, model) < 0:
        done = 1

    if sum(state[0]) <= threshold[0] or sum(state[1]) <= threshold[1]:
        done = 1
    return done

def findSimilarity(pair, pid, framesCount):
    id1 = pair[0][0]
    cam1 = pair[0][1]
    id2 = pair[1][0]
    cam2 = pair[1][1]

    pooledFeatureA = torch.zeros(1, 128)
    pooledFeatureB = torch.zeros(1, 128)
    frameFeaturesA = torch.load(dirPath + 'temporalRepresentation/' + str(cam1) + '/' + str(id1)+'.pt')
    frameFeaturesB = torch.load(dirPath + 'temporalRepresentation/' + str(cam2) + '/' + str(id2)+'.pt')

    for i in range(framesCount[0]):
        pooledFeatureA += frameFeaturesA[i] * weights[0][i]
    pooledFeatureA /= framesCount[0]

    for i in range(framesCount[1]):
        pooledFeatureB += frameFeaturesB[i] * weights[1][i]
    pooledFeatureB /= framesCount[1]

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarityAB = cos(pooledFeatureA, pooledFeatureB)
    return similarityAB
