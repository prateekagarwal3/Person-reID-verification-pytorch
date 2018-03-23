import prepareDataset

import os
import copy
import random
from itertools import count
from collections import namedtuple

import time
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

frameDropThreshold = 0.625
PRELU_WEIGHT = torch.FloatTensor([0.25])
if torch.cuda.is_available():
    PRELU_WEIGHT = PRELU_WEIGHT.cuda()
PRELU_WEIGHT = Variable(PRELU_WEIGHT)

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))

seqRootRGB = '/data/home/prateeka/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
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
    maxFrameCount = 192
    pid = torch.IntTensor([triplet[0], triplet[1], triplet[2]])
    fc1 = personFramesDict[personIdxDict[triplet[0]]][0]
    fc2 = personFramesDict[personIdxDict[triplet[1]]][1]
    fc3 = personFramesDict[personIdxDict[triplet[2]]][1]
    # print fc1, fc2, fc3
    fcMax = max(fc1, fc2, fc3)
    framesCount = torch.IntTensor([fc1, fc2, fc3])
    threshold = torch.IntTensor([int(fc1*frameDropThreshold), int(fc2*frameDropThreshold), int(fc3*frameDropThreshold)])
    initialState = torch.ones(3, maxFrameCount)
    initialState[0, fc1:maxFrameCount] = 0
    initialState[1, fc2:maxFrameCount] = 0
    initialState[2, fc3:maxFrameCount] = 0
    tempDict = {}
    tempDict['A'] = [i for i in range(0, fc1-4)]
    tempDict['B'] = [i for i in range(0, fc2-4)]
    tempDict['C'] = [i for i in range(0, fc3-4)]
    framesDropInfo = torch.IntTensor(3, maxFrameCount)
    framesDropInfo[0, 0:fc1-4] = torch.IntTensor(tempDict['A'])
    framesDropInfo[0, fc1-4:maxFrameCount] = -1
    framesDropInfo[1, 0:fc2-4] = torch.IntTensor(tempDict['B'])
    framesDropInfo[1, fc2-4:maxFrameCount] = -1
    framesDropInfo[2, 0:fc3-4] = torch.IntTensor(tempDict['C'])
    framesDropInfo[2, fc3-4:maxFrameCount] = -1

    if torch.cuda.is_available():
        pid = pid.cuda()
        framesDropInfo = framesDropInfo.cuda()
        framesCount = framesCount.cuda()
        threshold = threshold.cuda()
        initialState = initialState.cuda()
    return pid, framesDropInfo, framesCount, threshold, initialState

def saveTestTriplet(testTriplets):
    triplets = torch.IntTensor(75, 3)
    for i in range(75):
        triplets[i] = torch.IntTensor(testTriplets[i])
    torch.save(triplets, "testTriplets.pt")

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
        rootDir = "/data/home/prateeka/dataset/iLIDS-VID/i-LIDS-VID/sequences/"
        camDir = "cam"
        pidDir = ""
        # print pid_batch
        if int(action_batch[i][0]) is 0:
            camDir = camDir + "1"
            pidDir = personIdxDict[pid_batch[i][0].data[0]]
        elif int(action_batch[i][0]) is 1:
            camDir = camDir + "2"
            pidDir = personIdxDict[pid_batch[i][1].data[0]]
        elif int(action_batch[i][0]) is 2:
            camDir = camDir + "2"
            pidDir = personIdxDict[pid_batch[i][2].data[0]]

        rootDir = os.path.join(rootDir, camDir, pidDir)
        framesDropIndex = action_batch[i][1]
        frames_batch[i] = loadDroppedFrames(rootDir, framesDropIndex)
    if torch.cuda.is_available():
        frames_batch = frames_batch.cuda()
    return frames_batch

def findSimilarity(weights, pid, framesCount):

    pooledFeatureA = torch.zeros(1, 128)
    pooledFeatureB = torch.zeros(1, 128)
    pooledFeatureC = torch.zeros(1, 128)
    frameFeaturesA = torch.load('/data/home/prateeka/temporalRepresentation/cam1/' + str(pid[0])+'.pt')
    frameFeaturesB = torch.load('/data/home/prateeka/temporalRepresentation/cam2/' + str(pid[1])+'.pt')
    frameFeaturesC = torch.load('/data/home/prateeka/temporalRepresentation/cam2/' + str(pid[2])+'.pt')

    for i in range(framesCount[0]):
        pooledFeatureA += frameFeaturesA[i] * weights[0][i]
    pooledFeatureA /= framesCount[0]

    for i in range(framesCount[1]):
        pooledFeatureB += frameFeaturesB[i] * weights[1][i]
    pooledFeatureB /= framesCount[1]

    for i in range(framesCount[2]):
        pooledFeatureC += frameFeaturesC[i] * weights[2][i]
    pooledFeatureC /= framesCount[2]
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    similarityAB = cos(pooledFeatureA, pooledFeatureB)
    similarityAC = cos(pooledFeatureA, pooledFeatureC)
    return (similarityAB, similarityAC)

def findReward(weights, newWeights, pid, framesCount):
    initialSimilarityAB, initialSimilarityAC = findSimilarity(weights, pid, framesCount)
    # print initialSimilarityAB, initialSimilarityAC
    newSimilarityAB, newSimilarityAC = findSimilarity(newWeights, pid, framesCount)
    reward = (newSimilarityAB - initialSimilarityAB) - (newSimilarityAC - initialSimilarityAC)
    return reward

def getframeDropIndex(framesDropInfo, channel):
    a = framesDropInfo[channel]
    while True:
        # print a
        x = random.sample(a, 1)[0]
        if x == -1:
            x = random.sample(a, 1)[0]
        if x != -1:
            return x

def genAllAction(pid, framesCount, state, framesDropInfo):
    action_batch = torch.Tensor(400, 2)
    numActions = 0

    for c in range(0, 3):
        for index in framesDropInfo[c]:
            # print index
            if int(index) is -1:
                continue
            # if index == 0:
                # print("Channel for zero index" , c)
            action_batch[numActions] = torch.IntTensor([c, index.data[0]])
            numActions += 1
            tempS = state
            tempF = framesDropInfo.data
            nextState = state.clone()
            for i in range(0, 5):
                if i + index.data[0] < framesCount[c].data[0]:
                    temp = (i + index.data[0])
                    nextState[c, temp] = 0
            for i in range(0,5):
                if index.data[0]+i in tempF[c]:
                    temp = (index.data[0]+i)
                    tempF[c, temp] = -1
            for i in range(1,5):
                if index.data[0] - i in tempF[c]:
                    temp = (index.data[0]-i)
                    tempF[c, temp] = -1

    action_batch = action_batch[0:numActions]
    pid_batch = torch.Tensor(numActions, 3)
    framesCount_batch = torch.Tensor(numActions, 3)
    state_batch = torch.Tensor(numActions, 3, state.size(1))
    for i in range(numActions):
        pid_batch[i] = pid.data
        framesCount_batch[i] = framesCount.data
        state_batch[i] = state.data
    if torch.cuda.is_available():
        pid_batch = pid_batch.cuda()
        framesCount_batch = framesCount_batch.cuda()
        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
    return pid_batch, framesCount_batch, state_batch, action_batch

def getAction(pid, framesCount, state, framesDropInfo, model):
    if random.random() < 0.1:
        # print("Runnning get action")
        channel = random.randint(0, 2)
        action = torch.IntTensor(2)
        action[0] = channel
        action[1] = getframeDropIndex(framesDropInfo, channel)

        return action
    else:
        pid_batch, framesCount_batch, state_batch, action_batch = genAllAction(Variable(pid), Variable(framesCount), Variable(state), Variable(framesDropInfo))
        modelTic = time.time()
        bestActionIndex = model(Variable(pid_batch), Variable(framesCount_batch), Variable(state_batch), Variable(action_batch)).max(0)[1].data
        modelToc = time.time()
        print("Time taken in model action finder: {} seconds".format(modelToc-modelTic))

        # bestActionIndex.volatile = False
        return torch.IntTensor([int(action_batch[bestActionIndex][0][0]), int(action_batch[bestActionIndex][0][1])])

def checkTerminalState(state, threshold, pid, framesDropInfo, framesCount):
    # print(sum(state[0]), threshold[0])
    # print(sum(state[1]), threshold[1])
    # print(sum(state[2]), threshold[2])

    doneT = 1 if sum(state[0]) <= threshold[0] or sum(state[1]) <= threshold[1] or sum(state[2]) <= threshold[2] else 0
    # print("gfsgfsgsg", doneT)

    doneR = 1
    for c in range(0, 3):
        for index in framesDropInfo[c]:
            if index == -1:
                continue
            tempF = framesDropInfo.clone()
            nextState = state.clone()
            for i in range(0, 5):
                if i + index < framesCount[c]:
                    nextState[c, (i + index)] = 0
            for i in range(0,5):
                if index+i in tempF[c]:
                    tempF[c, (index+i)] = -1
            for i in range(1,5):
                if index - i in tempF[c]:
                    tempF[c, (index-i)] = -1
            reward = findReward(state, nextState, pid, framesCount)
            if reward[0] >= 0:
                doneR = 0
                return doneR or doneT
    return doneR or doneT

def performAction(state, action, threshold, pid, framesDropInfo, framesCount):
    nextState = state.clone()
    for i in range(0, 5):
        nextState[action[0], (i + action[1])] = 0
    for i in range(0,5):
        if action[1]+i in framesDropInfo[action[0]]:
            framesDropInfo[action[0], (action[1]+i)] = -1
    for i in range(1,5):
        if action[1] - i in framesDropInfo[action[0]]:
            framesDropInfo[action[0], (action[1]-i)] = -1

    done = 1 if checkTerminalState(state, threshold, pid, framesDropInfo, framesCount) else 0
    reward = findReward(state, nextState, pid, framesCount)
    if done:
        nextState[0][0] = -1
        nextState[0][1] = -1
        nextState[1][0] = -1
        nextState[1][1] = -1
        nextState[2][0] = -1
        nextState[2][1] = -1
    if torch.cuda.is_available():
        nextState = nextState.cuda()
    return nextState, framesDropInfo, reward, done

def getOrderStats(pid, framesCount, state, action):
    channel = int(action[0][0])
    personNo = int(pid[0][channel])
    if int(action[0][0] == 0):
        cam = 1
    else:
        cam = 2
    # print personNo
    personId = personIdxDict[personNo]
    frameFeatures = torch.load('/data/home/prateeka/temporalRepresentation/cam' + str(cam) + '/' + str(personNo)+'.pt')
    poolFeatures = torch.zeros(1, 128)
    for i in range(int(framesCount[0][channel])):
        poolFeatures += state[0][channel][i] * frameFeatures[i]
    poolDroppedFeatures = torch.zeros(1, 128)
    for i in range(int(action[0][1]), int(action[0][1])+5):
        poolDroppedFeatures += frameFeatures[i]
    poolDroppedFeatures /= 5
    poolFeatures -= poolDroppedFeatures
    v1 = poolFeatures
    v2 = torch.FloatTensor([torch.var(frameFeatures)]).view(1, 1)

    for i in range(1, pid.size(0)):
        channel = int(action[i][0])
        personNo = int(pid[i][channel])
        if int(action[i][0]) == 0:
            cam = 1
        else:
            cam = 2
        personId = personIdxDict[personNo]
        frameFeatures = torch.load('/data/home/prateeka/temporalRepresentation/cam' + str(cam) + '/' + str(personNo)+'.pt')
        poolFeatures = torch.zeros(1, 128)
        for j in range(int(framesCount[i][channel])):
            poolFeatures += state[i][channel][j] * frameFeatures[j]
        poolDroppedFeatures = torch.zeros(1, 128)
        for j in range(int(action[i][1]), int(action[i][1])+5):
            poolDroppedFeatures += frameFeatures[j]
        poolDroppedFeatures /= 5
        poolFeatures -= poolDroppedFeatures
        x = poolFeatures
        y = torch.FloatTensor([torch.var(frameFeatures)]).view(1, 1)
        v1 = torch.cat((v1, x), dim=0)
        v2 = torch.cat((v2, y), dim=0)

    if torch.cuda.is_available():
        v1 = v1.cuda()
        v2 = v2.cuda()
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
        if torch.cuda.is_available():
            # print "in Loop"
            x = x.cuda()
        x = Variable(x)
        # print x
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
