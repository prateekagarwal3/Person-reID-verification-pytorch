import utilsRL
import prepareDataset

import copy
import math
import random
from PIL import Image
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

torch.manual_seed(7)

testTrainSplit = 0.75
BATCH_SIZE = 8
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.11
EPS_DECAY = 200

Transition = namedtuple('Transition', ('pid', 'framesCount', 'state', 'action', 'nextState', 'reward', 'framesDropInfo'))

model = utilsRL.DQN()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.RMSprop(model.parameters(), lr = 1e-6)
memory = utilsRL.ReplayMemory(10000)
episodeDurations = []

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
trainTriplets, testTriplets = prepareDataset.generateTriplets(len(personFramesDict), testTrainSplit)

def optimizeModel(model):
    print("Optimizing Model Begin")
    if len(memory) < BATCH_SIZE:
        print("Optimizing Model End because of not sufficient training examples in replay memory")
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    pid_batch = Variable(torch.cat(batch.pid).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.pid))

    framesCount_batch = Variable(torch.cat(batch.framesCount).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.framesCount))

    state_batch = Variable(torch.cat(batch.state).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.state))

    action_batch = Variable(torch.cat(batch.action).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.action))

    reward_batch = Variable(torch.cat(batch.reward).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.reward))

    # print pid_batch, framesCount_batch, state_batch, action_batch

    stateActionValues = model(pid_batch, framesCount_batch, state_batch, action_batch)

    framesDropInfo_batch = Variable(torch.cat(batch.framesDropInfo).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.framesDropInfo))

    nextState_batch = Variable(torch.cat(batch.nextState).cuda()) if torch.cuda.is_available() else Variable(torch.cat(batch.nextState))

    nextStateValues = torch.Tensor(BATCH_SIZE, 1)
    for i in range(BATCH_SIZE):
        if int(nextState_batch[i][0][0]) == -1:
            nextStateValues[i] = 0
        else:
            pid_batch, framesCount_batch, state_batch, action_batch = utilsRL.generateAllAction(pid_batch[i], framesCount_batch[i], nextState_batch[i], framesDropInfo_batch[i])
            # print pid_batch, framesCount_batch, state_batch, action_batch
            input = pid_batch, framesCount_batch, state_batch, action_batch
            nextStateValues[i] = model(*input).max(0)[0].data

    nextStateValues.volatile = False
    expectedStateActionValues = (nextStateValues * GAMMA) + reward_batch.data.view(reward_batch.size(0), -1)

    loss = F.smooth_l1_loss(stateActionValues, Variable(expectedStateActionValues))

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            cparam.grad.data.clamp_(-0.5, 0.5)
    optimizer.step()

    print("Optimizing Model End")

temp = 1
for triplet in trainTriplets:
    temp += 1
    if temp > 2:
        break
    print("Triplet Loop Running, triplet=", triplet)
    triplet = [34, 34, 172]
    framesDropInfo, threshold, initialState = utilsRL.getTripletInfo(triplet, personIdxDict, personFramesDict)
    pid = {}
    pid['A'] = personIdxDict[triplet[0]]
    pid['B'] = personIdxDict[triplet[1]]
    pid['C'] = personIdxDict[triplet[2]]
    state = copy.deepcopy(initialState)

    for t in count():
        print("T Loop Running current t=", t)
        action, framesDropInfo = utilsRL.getAction(state, framesDropInfo)
        nextState, reward, done = utilsRL.performAction(state, action, threshold, pid, framesDropInfo)

        pidMem, framesCount, stateMem, actionMem, nextStateMem, framesDropInfoMem = utilsRL.dictToTensor(pid, state, action, nextState, framesDropInfo)

        memory.push(pidMem, framesCount, stateMem, actionMem, nextStateMem, reward, framesDropInfoMem)

        state = copy.deepcopy(nextState)
        optimizeModel(model)
        if done:
            episodeDurations.append(t + 1)
            break

torch.save(model.state_dict(), '/Users/prateek/8thSem/rl-person-verification/runs/model_run_dqn.pt')
