import buildModel
import prepareDataset

import os
import sys
import math
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

if sys.platform.startswith('linux'):
    dirPath = '/data/home/prateeka/'
elif sys.platform.startswith('darwin'):
    dirPath = '/Users/prateek/8thSem/'

seqRootRGB = dirPath + 'dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())
# print personIdxDict

torch.manual_seed(7)

sequence_length = 16
input_size = 64
hidden_size = 64
num_layers = 2
learning_rate = 0.001
momentum = 0
alpha = torch.FloatTensor([0.4])
if torch.cuda.is_available():
    alpha = alpha.cuda()
alpha = Variable(alpha)
num_epochs = 100
batch_size = 1
testTrainSplit = 0.75
steps_done = 0
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000
MAX_FRAME_COUNT = 192

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
    if fc1 < maxFrameCount:
        initialState[0, fc1:maxFrameCount] = 0
    if fc2 < maxFrameCount:
        initialState[1, fc2:maxFrameCount] = 0
    if fc3 < maxFrameCount:
        initialState[2, fc3:maxFrameCount] = 0

    tempDict = {}
    tempDict['A'] = [i for i in range(0, fc1-19)]
    tempDict['B'] = [i for i in range(0, fc2-19)]
    tempDict['C'] = [i for i in range(0, fc3-19)]
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
    for i in range(0, len(tempDict['C'])-4, 5):
        tempDict['C'][i+1] = -1
        tempDict['C'][i+2] = -1
        tempDict['C'][i+3] = -1
        tempDict['C'][i+4] = -1
    for i in range(0, len(tempDict['C'])):
        if i % 5 != 0 and tempDict['C'][i] != -1:
            tempDict['C'][i] = -1
    framesDropInfo = torch.IntTensor(3, maxFrameCount)
    framesDropInfo[0, 0:fc1-19] = torch.IntTensor(tempDict['A'])
    framesDropInfo[0, fc1-19:maxFrameCount] = -1
    framesDropInfo[1, 0:fc2-19] = torch.IntTensor(tempDict['B'])
    framesDropInfo[1, fc2-19:maxFrameCount] = -1
    framesDropInfo[2, 0:fc3-19] = torch.IntTensor(tempDict['C'])
    framesDropInfo[2, fc3-19:maxFrameCount] = -1

    if torch.cuda.is_available():
        pid = pid.cuda()
        framesDropInfo = framesDropInfo.cuda()
        framesCount = framesCount.cuda()
        threshold = threshold.cuda()
        initialState = initialState.cuda()
    return pid, framesDropInfo, framesCount, threshold, initialState

def saveTestTriplet(testTriplets):
    triplets = torch.IntTensor(len(testTriplets), 3)
    for i in range(len(testTriplets)):
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
    # printframeList
    if(frameList[0] == '.DS_Store'):
        frameList.remove('.DS_Store')
    # print("fdi", framesDropIndex)
    frameFileName = os.path.join(rootDir, frameList[framesDropIndex])
    x = loadImage(frameFileName)
    frameFileName = rootDir
    for i in range(framesDropIndex + 2, framesDropIndex + 20, 2):
        frameFileName = rootDir
        frameFileName = os.path.join(rootDir, frameList[i])
        y = loadImage(frameFileName)
        x = torch.cat([x, y], 0)
    return x

def generateFramesBatch(pid_batch, action_batch):
    frames_batch = torch.Tensor(pid_batch.size(0), 30, 128, 64)
    for i in range(0, pid_batch.size(0)):
        rootDir = dirPath + 'dataset/iLIDS-VID/i-LIDS-VID/sequences/'
        camDir = 'cam'
        pidDir = ''
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
    frameFeaturesA = torch.load(dirPath + 'temporalRepresentation/cam1/' + str(pid[0])+'.pt')
    frameFeaturesB = torch.load(dirPath + 'temporalRepresentation/cam2/' + str(pid[1])+'.pt')
    frameFeaturesC = torch.load(dirPath + 'temporalRepresentation/cam2/' + str(pid[2])+'.pt')

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

def countOnes(state):
    fc1 = 0
    fc2 = 0
    fc3 = 0
    for i in range(state.size(1)):
        if int(state[0][i]) is 1:
            fc1 += 1
    for i in range(state.size(1)):
        if int(state[1][i]) is 1:
            fc2 += 1
    for i in range(state.size(1)):
        if int(state[2][i]) is 1:
            fc3 += 1
    return fc1, fc2, fc3

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
        h0 = Variable(h0)
        c0 = Variable(c0)
        out, hidden_data = self.lstm(x, (h0, c0))
        return out

rnn = RNN(input_size, hidden_size, num_layers)
tripletRNNRGB = buildModel.TripletNet(rnn)
if torch.cuda.is_available():
    tripletRNNRGB.cuda()

tripletRNNOP = buildModel.TripletNet(rnn)
if torch.cuda.is_available():
    tripletRNNOP.cuda()

tripletRNNRGB.load_state_dict(torch.load(dirPath + 'gpu-rl/runs/model_run_rgb.pt'))
tripletRNNOP.load_state_dict(torch.load(dirPath + 'gpu-rl/runs/model_run_op.pt'))

def getState(pid, framesCount, state, framesDropInfo):
    fc1, fc2, fc3 = countOnes(state)
    if fc1 < sequence_length or fc2 < sequence_length or fc3 < sequence_length :
        return torch.randn(3, 128).cuda() if torch.cuda.is_available() else torch.randn(3, 128)
    # for j in range(sequence_length - fc1):
    #     for i in range(state[0].size(0)):
    #         if state[0, i] is not 1:
    #             state[0, i] = 1
    #             break
    # for j in range(sequence_length - fc2):
    #     for i in range(state[1].size(0)):
    #         if state[1, i] is not 1:
    #             state[1, i] = 1
    #             break
    # for j in range(sequence_length - fc3):
    #     for i in range(state[2].size(0)):
    #         if state[2, i] is not 1:
    #             state[2, i] = 1
    #             break
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
            # print("pid", pid)
            # print("frame size", personFramesDict[personIdxDict[256]])
            # print("anchor size", anchorFrames.size())
            # print("src size", anchorIP[i].size())
            # print("dest size", anchorFrames[0 : sequence_length].size())
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

def getStateValues(pid_batch, framesCount_batch, state_batch, framesDropInfo_batch, model):
    values = torch.Tensor(pid_batch.size(0) / 3, 1)
    for x in range(0, pid_batch.size(0), 3):
        pid = pid_batch[x:x+3][:]
        # print pid_batch.size()
        framesCount = framesCount_batch[x:x+3][:]
        state = state_batch[x:x+3][:]
        framesDropInfo = framesDropInfo_batch[x:x+3][:]
        # print framesDropInfo_batch.size()
        stateReduced = getState(pid, framesCount, state, framesDropInfo)
        valueTic = time.time()
        qValues = model(stateReduced).data
        # print("qvales", torch.max(qValues))
        valueToc = time.time()
        # print("Time taken by value finder : {}seconds".format( valueToc-valueTic))
        # print qValues
        maxValue = qValues[0][0]
        maxIndex = torch.IntTensor([0, 0])
        # print("printing qvalues size", qValues.size())
        for i in range(qValues.size(0)):
            for j in range(qValues.size(1)):
                if j * 5 < int(framesCount[i]) and int(framesDropInfo[i][j*5]) != -1:
                    if maxValue < qValues[i][j]:
                        # print("loop running")
                        maxValue = qValues[i][j]
                        # print maxValue
                        # print qValues[i][j]
                        maxIndex = torch.IntTensor([i, j*5])
        values[x / 3] = maxValue
    if torch.cuda.is_available():
        values = values.cuda()
    # print values
    return Variable(values)

def getframeDropIndex(framesDropInfo, channel):
    a = framesDropInfo[channel]
    while True:
        # print a
        x = random.sample(a, 1)[0]
        # print "x", x
        if int(torch.max(a)) is -1:
            return -1
        if x == -1:
            x = random.sample(a, 1)[0]
            # print("runugfsgfggf")
        if x != -1:
            return x

def getAction(pid, framesCount, state, framesDropInfo, policy_net):
    global steps_done
    done = 0
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # print("Best action taken")
        stateReduced = getState(pid, framesCount, state, framesDropInfo)
        # print stateReduced
        qValues = policy_net(stateReduced).data
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
        return done, maxIndex
    else:
        done = 0
        action = torch.IntTensor(2)
        tempC = random.randint(0, 2)
        tempA = getframeDropIndex(framesDropInfo, tempC)
        checkChannel = [0, 0, 0]
        checkChannel[tempC] = 1
        while(True):
            if tempA == -1:
                tempC = random.randint(0, 2)
                tempA = getframeDropIndex(framesDropInfo, tempC)
                checkChannel[tempC] = 1
            elif tempA != -1:
                break
            elif sum(checkChannel) == 3 and tempA == -1:
                print "No Action possible"
                done = 1
                break
        action[0] = tempC
        action[1] = tempA
        return done, action

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
    for i in range(0, 10, 2):
        nextState[action[0], (i + action[1])] = 0
    for i in range(0, 10, 2):
        if action[1]+i in framesDropInfo[action[0]]:
            framesDropInfo[action[0], (action[1]+i)] = -1
    for i in range(1, 10, 2):
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

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 37)
        # self.dp1 = nn.Dropout(p=0.5)
        # self.dp2 = nn.Dropout(p=0.25)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        # print x.size()
        x = Variable(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        # print xc
        return x

if __name__ == "__main__":
    print("Main is running")
