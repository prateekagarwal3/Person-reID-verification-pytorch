import utilsRL
import prepareDataset

import sys
import time
import copy
import math
import random
from PIL import Image
from itertools import count
from torchviz import make_dot
from collections import namedtuple
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

torch.manual_seed(7)

num_epochs = 100
testTrainSplit = 0.75
BATCH_SIZE = 8
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))

if sys.platform.startswith('linux'):
    dirPath = '/data/home/prateeka/'
elif sys.platform.startswith('darwin'):
    dirPath = '/Users/prateek/8thSem/'

model = utilsRL.DQN()
if torch.cuda.is_available():
    model = model.cuda()

torch.backends.cudnn.enabled = True

optimizer = optim.RMSprop(model.parameters(), lr = 1e-6)
memory = utilsRL.ReplayMemory(10000)
episodeDurations = []

seqRootRGB = dirPath + 'dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
trainTriplets, testTriplets = prepareDataset.generateTriplets(len(personFramesDict), testTrainSplit)
personNoDict = dict([v,k] for k,v in personIdxDict.items())
utilsRL.saveTestTriplet(testTriplets)

def optimizeModel(model):
    print("Optimizing Model Begin")
    if len(memory) < BATCH_SIZE:
        print("Optimizing Model End because of not sufficient training examples in replay memory")
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # print batch

    pid_batch = Variable(torch.cat(batch.pid).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.pid))
    pid_batch = pid_batch.view(BATCH_SIZE, -1)

    framesCount_batch = Variable(torch.cat(batch.framesCount).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.framesCount))
    framesCount_batch = framesCount_batch.view(BATCH_SIZE, -1)
    # print batch.state
    state_batch = Variable(torch.cat(batch.state).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.state))
    state_batch = state_batch.view(BATCH_SIZE, 3, -1)
    # print state_batch.size()

    action_batch = Variable(torch.cat(batch.action).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.action))
    action_batch = action_batch.view(BATCH_SIZE, -1)

    reward_batch = Variable(torch.cat(batch.reward).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.reward))
    reward_batch = reward_batch.view(BATCH_SIZE, -1)

    stateActionValues = model(pid_batch, framesCount_batch, state_batch, action_batch)
    framesDropInfo_batch = Variable(torch.cat(batch.framesDropInfo).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.framesDropInfo))
    framesDropInfo_batch = framesDropInfo_batch.view(BATCH_SIZE, 3, -1)

    nextState_batch = Variable(torch.cat(batch.nextState).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.nextState))
    nextState_batch = nextState_batch.view(BATCH_SIZE, 3, -1)
    nextStateValues = torch.Tensor(BATCH_SIZE, 1)

    if torch.cuda.is_available():
        nextStateValues = nextStateValues.cuda()

    pidBatch_List = []
    framesCountBatch_List = []
    stateBatch_List = []
    actionBatch_List = []

    for i in range(BATCH_SIZE):
        pid_tbatch, framesCount_tbatch, state_tbatch, action_tbatch = utilsRL.genAllAction(pid_batch[i], framesCount_batch[i], nextState_batch[i], framesDropInfo_batch[i])
        pidBatch_List.append(pid_tbatch)
        framesCountBatch_List.append(framesCount_tbatch)
        stateBatch_List.append(state_tbatch)
        actionBatch_List.append(action_tbatch)

    pid_nbatch = torch.cat(pidBatch_List, dim=0)
    framesCount_nbatch = torch.cat(framesCountBatch_List, dim=0)
    state_nbatch = torch.cat(stateBatch_List, dim=0)
    action_nbatch = torch.cat(actionBatch_List, dim=0)
    b = []
    for i in range(BATCH_SIZE):
        b.append(pid_nbatch[i].size(0))
    for i in range(1, BATCH_SIZE):
        b[i] += b[i-1]
    print("input size printing", pid_nbatch.size(0))
    input = Variable(pid_nbatch), Variable(framesCount_nbatch), Variable(state_nbatch), Variable(action_nbatch)
    modelTic = time.time()
    output = model(*input)
    modelToc = time.time()
    print("Time taken in model inference:{}seconds".format(modelToc-modelTic))
    # output.volatile = False
    nextStateValues.volatile = False
    for i in range(BATCH_SIZE-1):
        if nextState_batch[i][0][0].data[0] == 0:
            nextStateValues[i] = 0
        else:
            nextStateValues[i] = output[b[i]: b[i+1]].max(0)[0].data
    expectedStateActionValues = (nextStateValues * GAMMA) + reward_batch.data.view(reward_batch.size(0), -1)

    loss = F.smooth_l1_loss(stateActionValues, Variable(expectedStateActionValues))
    lossTic = time.time()
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-0.5, 0.5)
    optimizer.step()
    lossToc = time.time()
    print("Optimizing Model End")

temp = 1
for epoch in range(num_epochs):
    epochTic = time.time()
    for triplet in trainTriplets:
        tripletTic = time.time()
        # temp += 1
        # if temp > 2:
        #     break
        # # triplet = [34, 34, 172]
        # triplet = [259, 259, 133]
        print("Triplet Loop Running, triplet=", triplet)
        pid, framesDropInfo, framesCount, threshold, initialState = utilsRL.getTripletInfo(triplet, personIdxDict, personFramesDict)
        # print framesDropInfo
        state = initialState.clone()
        for t in count():
            # print("initialState", state)
            print("T Loop Running current t=", t)
            # print framesDropInfo.view(3, 81)
            action = utilsRL.getAction(pid.clone(), framesCount.clone(), state.clone(), framesDropInfo.clone(), model)
            # print("initialState", state)
            nextState, framesDropInfo, reward, done = utilsRL.performAction(state, action, threshold, pid, framesDropInfo, framesCount)
            memory.push(pid, framesCount, state, action, nextState, reward, framesDropInfo)
            # print("nextState", nextState)
            state = nextState.clone()
            print(sum(state[0]), threshold[0])
            print(sum(state[1]), threshold[1])
            print(sum(state[2]), threshold[2])
            # print done
            optimizeModel(model)
            # print("done", done)
            if done:
                episodeDurations.append(t + 1)
                break

        tripletToc = time.time()
        print("Time taken by triplet:{}seconds".format( tripletToc-tripletTic))
    epochToc = time.time()
    print("Time by Epoch: {}/{} {}seconds".format(epoch, num_epochs-1, epochToc - epochTic))

# print episodeDurations
# print memory.__len__()
torch.save(model.state_dict(), dirPath + 'rl-person-verification/runs/model_run_dqn.pt')
