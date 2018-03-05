import copy
import random
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

frameDropThreshold = 0.625

Transition = namedtuple('Transition', ('pid', 'state', 'action', 'next_state', 'reward'))

def getTripletInfo(triplet, personIdxDict, personFramesDict):
    pid = {}
    pid['A'] = personIdxDict[triplet[0]]
    pid['B'] = personIdxDict[triplet[1]]
    pid['C'] = personIdxDict[triplet[2]]
    framesCount = {}
    framesCount['A'] = personFramesDict[pid['A']][0]
    framesCount['B'] = personFramesDict[pid['B']][1]
    framesCount['C'] = personFramesDict[pid['C']][1]
    threshold = {}
    threshold['A'] = int(framesCount['A'] * frameDropThreshold)
    threshold['B'] = int(framesCount['B'] * frameDropThreshold)
    threshold['C'] = int(framesCount['C'] * frameDropThreshold)
    initialState = {}
    initialState['A'] = [[1]*framesCount['A']][0]
    initialState['B'] = [[1]*framesCount['B']][0]
    initialState['C'] = [[1]*framesCount['C']][0]
    framesDropInfo = {}
    framesDropInfo['A'] = [i for i in range(0, framesCount['A']-4)]
    framesDropInfo['B'] = [i for i in range(0, framesCount['B']-4)]
    framesDropInfo['C'] = [i for i in range(0, framesCount['C']-4)]

    return framesDropInfo, threshold, initialState

def modifyTriplets(trainTriplets, testTriplets, personIdxDict):
    for i in range(len(trainTriplets)):
        trainTriplets[i][0] = personIdxDict[trainTriplets[i][0]]
        trainTriplets[i][1] = personIdxDict[trainTriplets[i][1]]
        trainTriplets[i][2] = personIdxDict[trainTriplets[i][2]]
    for i in range(len(testTriplets)):
        testTriplets[i][0] = personIdxDict[testTriplets[i][0]]
        testTriplets[i][1] = personIdxDict[testTriplets[i][1]]
        testTriplets[i][2] = personIdxDict[testTriplets[i][2]]
    return trainTriplets, testTriplets

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def displayMemory(self):
        print self.memory

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=9)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.mp = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(6400, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.prelu(self.mp(self.conv1(x)))
        x = F.prelu(self.mp(self.conv2(x)))
        x = F.prelu(self.mp(self.conv3(x)))
        x = self.fc1(x)
        ''' Add v here'''
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def featureExtractor(frames, pid):
    a = torch.randn(len(frames['A']), 128)
    b = torch.randn(len(frames['B']), 128)
    c = torch.randn(len(frames['C']), 128)
    return a, b, c

def findSimilarity(weights, pid):
    weightsA = weights['A']
    weightsB = weights['B']
    weightsC = weights['C']
    framesCountA = len(weightsA)
    framesCountB = len(weightsB)
    framesCountC = len(weightsC)

    pooledFeatureA = torch.zeros(1, 128)
    pooledFeatureB = torch.zeros(1, 128)
    pooledFeatureC = torch.zeros(1, 128)
    frameFeaturesA, frameFeaturesB, frameFeaturesC = featureExtractor(weights, pid)

    for i in range(framesCountA):
        pooledFeatureA += frameFeaturesA[i] * weightsA[i]
    pooledFeatureA /= framesCountA

    for i in range(framesCountB):
        pooledFeatureB += frameFeaturesB[i] * weightsB[i]
    pooledFeatureA /= framesCountA

    for i in range(framesCountC):
        pooledFeatureC += frameFeaturesC[i] * weightsC[i]
    pooledFeatureC /= framesCountC
    # print pooledFeatureA, pooledFeatureB, pooledFeatureC
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarityAB = cos(pooledFeatureA, pooledFeatureB)
    similarityAC = cos(pooledFeatureA, pooledFeatureC)
    return (similarityAB, similarityAC)

def findReward(weights, newWeights, pid):
    initialSimilarityAB, initialSimilarityAC = findSimilarity(weights, pid)
    # print initialSimilarityAB, initialSimilarityAC
    newSimilarityAB, newSimilarityAC = findSimilarity(newWeights, pid)
    reward = (newSimilarityAB - initialSimilarityAB) - (newSimilarityAC - initialSimilarityAC)
    return reward

def getframeDropIndex(framesDropInfo, channel):
    index = random.sample(framesDropInfo[channel], 1)
    index = index[0]
    # print("Removing at index ", index)
    for i in range(0,5):
        if index + i in framesDropInfo[channel]:
            framesDropInfo[channel].remove(index + i)

    for i in range(1,5):
        if index - i in framesDropInfo[channel]:
            framesDropInfo[channel].remove(index-i)

    return index, framesDropInfo

def getAction(state, framesDropInfo):
    # print('Running getAction')
    channels = ['A', 'B', 'C']
    channel = random.sample(channels, 1)
    channel = channel[0]
    action = []
    action.append(channel)
    frameIdx, framesDropInfo = getframeDropIndex(framesDropInfo, channel)
    action.append(frameIdx)
    return action, framesDropInfo

def checkTerminalState(state, threshold, pid, framesDropInfo):
    doneT = 1 if sum(state['A']) <= threshold['A'] or sum(state['B']) <= threshold['B'] or sum(state['C']) <= threshold['C'] else 0
    doneR = 1
    return doneT

    for channel in ['A', 'B', 'C']:
        for index in framesDropInfo[channel]:
            tempS = copy.deepcopy(state)
            tempF = copy.deepcopy(framesDropInfo)
            nextState = copy.deepcopy(state)
            for i in range(0, 5):
                if i + index < len(state[channel]):
                    nextState[channel][i + index] = 0
            for i in range(0,5):
                if index+i in tempF[channel]:
                        tempF[channel].remove(index+i)
            for i in range(1,5):
                if index - i in tempF[channel]:
                    tempF[channel].remove(index-i)

            reward = findReward(state, nextState, pid)
            if reward >= 0:
                doneR = 0
                return doneR and doneT

    return doneR and doneT

def performAction(state, action, threshold, pid, framesDropInfo):
    nextState = state
    for i in range(0, 5):
        nextState[action[0]][i + action[1]] = 0
    # done = 1 if sum(state['A']) <= threshold['A'] and sum(state['A']) <= threshold['A'] and sum(state['C']) <= threshold['C'] else 0
    done = 1 if checkTerminalState(state, threshold, pid, framesDropInfo) else 0
    reward = findReward(state, nextState, pid)
    return nextState, reward, done
