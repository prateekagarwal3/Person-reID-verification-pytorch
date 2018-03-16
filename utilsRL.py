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
PRELU_WEIGHT = Variable(torch.FloatTensor([0.25]))

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())
# print personIdxDict

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
    # print len(initialState['A']), len(initialState['B']), len(initialState['C'])
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

def loadImage(filename):
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
    img = transforms.ToTensor()(img)
    if torch.cuda.is_available():
        img = img.cuda()
    return img

def loadDroppedFrames(rootDir, framesDropIndex):
    framesDropIndex = int(framesDropIndex.data[0])
    frameList = sorted(os.listdir(rootDir))
    if(frameList[0] == '.DS_Store'):
        frameList.remove('.DS_Store')

    frameFileName = os.path.join(rootDir, frameList[framesDropIndex])
    x = loadImage(frameFileName)
    frameFileName = rootDir
    for i in range(framesDropIndex + 1, framesDropIndex + 5):
        frameFileName = rootDir
        frameFileName = os.path.join(rootDir, frameList[i])
        y = loadImage(frameFileName)
        x = torch.cat([x, y], 0)
    return x

def generateFramesBatch(pid_batch, action_batch):
    frames_batch = torch.Tensor(pid_batch.size(0), 15, 128, 64)
    for i in range(0, pid_batch.size(0)):
        rootDir = "/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/"
        camDir = "cam"
        pidDir = ""
        # print action_batch
        if action_batch[i][0] == 0:
            camDir = camDir + "1"
            pidDir = personIdxDict[pid_batch[i][0].data[0]]
        elif action_batch[i][0] == 1:
            camDir = camDir + "2"
            pidDir = personIdxDict[pid_batch[i][1].data[0]]
        elif action_batch[i][0] == 2:
            camDir = camDir + "2"
            pidDir = personIdxDict[pid_batch[i][2].data[0]]

        rootDir = os.path.join(rootDir, camDir, pidDir)
        framesDropIndex = action_batch[i][1]
        frames_batch[i] = loadDroppedFrames(rootDir, framesDropIndex)

    return frames_batch

def featureExtractor(frames, pid):
    a = torch.randn(len(frames['A']), 256)
    b = torch.randn(len(frames['B']), 256)
    c = torch.randn(len(frames['C']), 256)
    return a, b, c

def dictToTensor(pid, state, action, nextState, framesDropInfo):
    framesCountA = len(state['A'])
    framesCountB = len(state['B'])
    framesCountC = len(state['C'])
    framesCount = torch.IntTensor([framesCountA, framesCountB, framesCountC])
    # print framesCount

    maxFramesCount = max(framesCountA, framesCountB, framesCountC)
    stateTorch = torch.ByteTensor(3, maxFramesCount)
    stateTemp = copy.deepcopy(state)
    for channel in ['A', 'B', 'C ']:
        channel = channel[0]
        if len(stateTemp[channel]) < maxFramesCount:
            for i in range(maxFramesCount - len(stateTemp[channel])):
                stateTemp[channel].append(0)

    stateTorch[0] = torch.IntTensor(stateTemp['A'])
    stateTorch[1] = torch.IntTensor(stateTemp['B'])
    stateTorch[2] = torch.IntTensor(stateTemp['C'])

    if nextState != None:
        nextStateTemp = copy.deepcopy(nextState)
        nextStateTorch = torch.ByteTensor(3, maxFramesCount)

        for channel in ['A', 'B', 'C ']:
            channel = channel[0]
            if len(nextStateTemp[channel]) < maxFramesCount:
                for i in range(maxFramesCount - len(nextStateTemp[channel])):
                    nextStateTemp[channel].append(0)

        nextStateTorch[0] = torch.IntTensor(nextStateTemp['A'])
        nextStateTorch[1] = torch.IntTensor(nextStateTemp['B'])
        nextStateTorch[2] = torch.IntTensor(nextStateTemp['C'])
        nextStateTorch = nextStateTorch.unsqueeze(0)
    else:
        nextStateTorch = None

    maxFramesDropInfo = max(len(framesDropInfo['A']), len(framesDropInfo['B']), len(framesDropInfo['C']))
    framesDropInfoTorch = torch.IntTensor(3, maxFramesDropInfo)

    for channel in ['A', 'B', 'C ']:
        channel = channel[0]
        if len(framesDropInfo[channel]) < maxFramesDropInfo:
            for i in range(maxFramesDropInfo - len(framesDropInfo[channel])):
                framesDropInfo[channel].append(0)

    framesDropInfoTorch[0] = torch.IntTensor(framesDropInfo['A'])
    framesDropInfoTorch[1] = torch.IntTensor(framesDropInfo['B'])
    framesDropInfoTorch[2] = torch.IntTensor(framesDropInfo['C'])

    if action[0] == 'A':
        actionTorch = torch.IntTensor([0, action[1]])
    if action[0] == 'B':
        actionTorch = torch.IntTensor([1, action[1]])
    if action[0] == 'C':
        actionTorch = torch.IntTensor([2, action[1]])
    # print("torch", actionTorch)

    pidTorch = torch.IntTensor([personNoDict[pid['A']], personNoDict[pid['B']], personNoDict[pid['C']]])

    return pidTorch.unsqueeze(0), framesCount.unsqueeze(0), stateTorch.unsqueeze(0), actionTorch.unsqueeze(0), nextStateTorch, framesDropInfoTorch.unsqueeze(0)

def findSimilarity(weights, pid):
    weightsA = weights['A']
    weightsB = weights['B']
    weightsC = weights['C']
    framesCountA = len(weightsA)
    framesCountB = len(weightsB)
    framesCountC = len(weightsC)

    pooledFeatureA = torch.zeros(1, 256)
    pooledFeatureB = torch.zeros(1, 256)
    pooledFeatureC = torch.zeros(1, 256)
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
    done = 1 if checkTerminalState(state, threshold, pid, framesDropInfo) else 0
    reward = findReward(state, nextState, pid)
    if done:
        nextState['A'][0] = -1
        nextState['A'][1] = -1
        nextState['B'][0] = -1
        nextState['B'][1] = -1
        nextState['C'][0] = -1
        nextState['C'][1] = -1

    return nextState, reward, done

def tensorToDict(framesCount, state, framesDropInfo):
    stateDict = {}
    for i in range(3):
        stateDict[chr(i+65)] = []
        for j in range(int(framesCount[i])):
            stateDict[chr(i+65)].append(state[i][j].data[0])
    # print stateDict

    framesDropInfoDict = {}
    for i in range(0, 3):
        count = 0
        for j in range(framesDropInfo[i].size(0)-1, -1, -1):
            count += 1
            if(framesDropInfo[i][j] > 0):
                break
        count = framesDropInfo[i].size(0) - count
        framesDropInfoDict[chr(i+65)] = []
        for j in range(count):
            framesDropInfoDict[chr(i+65)].append(framesDropInfo[i][j].data[0])
    # print framesDropInfoDict
    return stateDict, framesDropInfoDict

def generateAllAction(pid, framesCount, state, framesDropInfo):

    # print pid, framesCount, state, framesDropInfo
    action_batch = torch.Tensor(400, 2)
    numActions = 0
    stateDict, framesDropInfoDict = tensorToDict(framesCount, state, framesDropInfo)
    # print stateDict, framesDropInfoDict

    for channel in ['A', 'B', 'C']:
        # print "check"
        for index in framesDropInfoDict[channel]:
            tIndex = index
            action_batch[numActions] = torch.IntTensor([ord(channel)-65, tIndex])
            numActions += 1
            # print numActions
            tempS = copy.deepcopy(stateDict)
            tempF = copy.deepcopy(framesDropInfoDict)
            # print tempF
            nextState = copy.deepcopy(stateDict)
            for i in range(0, 5):
                if i + index < len(stateDict[channel]):
                    nextState[channel][i + tIndex] = 0
            for i in range(0,5):
                if tIndex+i in tempF[channel]:
                        tempF[channel].remove(tIndex+i)
            for i in range(1,5):
                if index - i in tempF[channel]:
                    tempF[channel].remove(tIndex-i)

    action_batch = action_batch[0:numActions]
    pid_batch = torch.Tensor(numActions, 3)
    framesCount_batch = torch.Tensor(numActions, 3)
    state_batch = torch.Tensor(numActions, 3, state.size(1))
    for i in range(numActions):
        pid_batch[i] = pid.data
        framesCount_batch[i] = framesCount.data
        state_batch[i] = state.data

    return Variable(pid_batch), Variable(framesCount_batch), Variable(state_batch), Variable(action_batch)

def getOrderStats(pid, framesCount, state, action):
    # print("fc", framesCount[action[0][0]])
    channel = int(action[0][0])
    personNo = pid[0][channel]
    # print personNo
    personId = personIdxDict[personNo]
    frameFeatures = torch.randn(int(framesCount[channel][0]), 128)
    poolFeatures = torch.zeros(1, 128)
    for i in range(int(framesCount[channel][0])):
        poolFeatures += state[0][channel][i] * frameFeatures[i]
    poolDroppedFeatures = torch.zeros(1, 128)
    for i in range(int(action[0][1]), 5):
        poolDroppedFeatures += frameFeatures[i]
    poolDroppedFeatures /= 5
    poolFeatures -= poolDroppedFeatures
    v1 = poolFeatures
    v2 = torch.FloatTensor([torch.var(frameFeatures)]).view(1, 1)

    for i in range(1, pid.size(0)):
        channel = int(action[i][0])
        personNo = pid[i][channel]
        personId = personIdxDict[personNo]
        frameFeatures = torch.randn(int(framesCount[channel][0]), 128)
        poolFeatures = torch.zeros(1, 128)
        for j in range(int(framesCount[channel][0])):
            poolFeatures += state[i][channel][j] * frameFeatures[j]
        poolDroppedFeatures = torch.zeros(1, 128)
        for j in range(int(action[i][1]), 5):
            poolDroppedFeatures += frameFeatures[j]
        poolDroppedFeatures /= 5
        poolFeatures -= poolDroppedFeatures
        x = poolFeatures
        y = torch.FloatTensor([torch.var(frameFeatures)]).view(1, 1)
        v1 = torch.cat((v1, x), dim=0)
        v2 = torch.cat((v2, y), dim=0)

    return v1, v2

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(15, 16, kernel_size=9)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3)
        self.mp = nn.MaxPool2d(4, stride=2)
        self.fc1 = nn.Linear(528, 128)
        self.fc2 = nn.Linear(257, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, pid, framesCount, state, action):
        # print action
        # print("pid", pid.size())
        x = generateFramesBatch(pid, action)
        x = Variable(x)
        x = F.prelu(self.mp(self.conv1(x)), weight=PRELU_WEIGHT)
        x = F.prelu(self.mp(self.conv2(x)), weight=PRELU_WEIGHT)
        x = F.prelu(self.mp(self.conv3(x)), weight=PRELU_WEIGHT)
        x = self.fc1(x.view(x.size(0), -1))
        v1, v2 = getOrderStats(pid.data, framesCount.data, state.data, action.data)
        # print("x", x.size())
        # print v1.size(), v2.size()
        x = Variable(torch.cat([x.data, v1, v2], dim=1))
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    print("Main is running")
