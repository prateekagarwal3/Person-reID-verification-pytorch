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

def generatePairs(testTriplets):
    
def checkTerminalState(state, threshold, pid, framesDropInfo):
    doneT = 1 if sum(state['A']) <= threshold['A'] or sum(state['B']) <= threshold['B'] or sum(state['C']) <= threshold['C'] else 0
    return doneR

def performAction(state, action):
    nextState = state
    for i in range(0, 5):
        nextState[action[0]][i + action[1]] = 0

def dictToTensor(pid, state, framesDropInfo):
    framesCountA = len(state['A'])
    framesCountB = len(state['B'])
    framesCount = torch.IntTensor([framesCountA, framesCountB])
    # print framesCount

    maxFramesCount = max(framesCountA, framesCountB)
    stateTorch = torch.ByteTensor(2, maxFramesCount)
    stateTemp = copy.deepcopy(state)
    for channel in ['A', 'B']:
        channel = channel[0]
        if len(stateTemp[channel]) < maxFramesCount:
            for i in range(maxFramesCount - len(stateTemp[channel])):
                stateTemp[channel].append(0)

    stateTorch[0] = torch.IntTensor(stateTemp['A'])
    stateTorch[1] = torch.IntTensor(stateTemp['B'])

    maxFramesDropInfo = max(len(framesDropInfo['A']), len(framesDropInfo['B'])))
    framesDropInfoTorch = torch.IntTensor(2, maxFramesDropInfo)

    for channel in ['A', 'B']:
        channel = channel[0]
        if len(framesDropInfo[channel]) < maxFramesDropInfo:
            for i in range(maxFramesDropInfo - len(framesDropInfo[channel])):
                framesDropInfo[channel].append(0)

    framesDropInfoTorch[0] = torch.IntTensor(framesDropInfo['A'])
    framesDropInfoTorch[1] = torch.IntTensor(framesDropInfo['B'])

    pidTorch = torch.IntTensor([personNoDict[pid['A']], personNoDict[pid['B']]



    )

    return pidTorch.unsqueeze(0), framesCount.unsqueeze(0), stateTorch.unsqueeze(0), actionTorch.unsqueeze(0), nextStateTorch, framesDropInfoTorch.unsqueeze(0)

def tensorToDict(framesCount, state, framesDropInfo):
    stateDict = {}
    for i in range(2):
        stateDict[chr(i+65)] = []
        for j in range(int(framesCount[i])):
            stateDict[chr(i+65)].append(state[i][j].data[0])
    # print stateDict

    framesDropInfoDict = {}
    for i in range(0, 2):
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

    for channel in ['A', 'B']:
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
    pid_batch = torch.Tensor(numActions, 2)
    framesCount_batch = torch.Tensor(numActions, 2)
    state_batch = torch.Tensor(numActions, 2, state.size(1))
    for i in range(numActions):
        pid_batch[i] = pid.data
        framesCount_batch[i] = framesCount.data
        state_batch[i] = state.data

    return Variable(pid_batch), Variable(framesCount_batch), Variable(state_batch), Variable(action_batch)
