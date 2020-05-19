import buildModel
import prepareDataset

import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

torch.manual_seed(7)

sequence_length = 16
input_size = 128
hidden_size = 128
num_layers = 2
learning_rate = 0.001
momentum = 0
alpha = torch.FloatTensor([0.4])
if torch.cuda.is_available():
    alpha = alpha.cuda()
alpha = Variable(alpha)
num_epochs = 2
batch_size = 1
testTrainSplit = 0.8

seqRootRGB = '/Users/prateek/8thSem/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
nTotalPersons = len(personFramesDict)
trainTriplets, testTriplets = prepareDataset.generateTriplets(nTotalPersons, testTrainSplit)

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

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
optimizerRGB = torch.optim.SGD(tripletRNNRGB.parameters(), lr=learning_rate, momentum=momentum)
optimizerOP = torch.optim.SGD(tripletRNNOP.parameters(), lr=learning_rate, momentum=momentum)

# if torch.cuda.is_available():
torch.backends.cudnn.benchmark = True
count = 1
tripletRNNRGB.train()
for epoch in range(num_epochs):
    # print("Epoch Loop Running", epoch)
    for triplet in trainTriplets:
        triplet = [15,15,213]
        count += 1
        if(count > 2):
            break
        # print("Triplet being used : ", triplet)
        for cam in range(1,2):
            #print("Cam Loop Running")
            anchor = personIdxDict[triplet[0]]
            positive = personIdxDict[triplet[1]]
            negative = personIdxDict[triplet[2]]

            framesCountList = personFramesDict[anchor]
            anchorFramesCount = framesCountList[cam-1]
            framesCountList = personFramesDict[positive]
            positiveFramesCount = framesCountList[2-cam]
            framesCountList = personFramesDict[negative]
            negativeFramesCount = framesCountList[2-cam]
            actualFramesCount = max(anchorFramesCount, positiveFramesCount, negativeFramesCount)
            # print(anchorFramesCount, positiveFramesCount, negativeFramesCount)

            anchorFrames = torch.randn(1, anchorFramesCount, 128)
            positiveFrames = torch.randn(1, positiveFramesCount, 128)
            negativeFrames = torch.randn(1, negativeFramesCount, 128)
            # anchorFrames = prepareDataset.getPersonFrames(seqRootRGB, anchor, cam, anchorFramesCount)
            # positiveFrames = prepareDataset.getPersonFrames(seqRootRGB, positive, 3-cam, positiveFramesCount)
            # negativeFrames = prepareDataset.getPersonFrames(seqRootRGB, negative, 3-cam, negativeFramesCount)

            quotient = actualFramesCount / sequence_length
            remainder = actualFramesCount % sequence_length

            anchorFramesRemaining = anchorFramesCount
            positiveFramesRemaining = positiveFramesCount
            negativeFramesRemaining = negativeFramesCount

            tempAnchorFramesRNNInput = anchorFrames[0][0:16]
            tempPositiveFramesRNNInput = positiveFrames[0][0:16]
            tempNegativeFramesRNNInput = negativeFrames[0][0:16]

            # print("Q, R:", quotient, remainder)
            for i in range(quotient+1):
                # print("Quotient Loop Running")
                if anchorFramesRemaining < 16:
                    anchorFramesRNNInput = tempAnchorFramesRNNInput
                else:
                    anchorFramesRNNInput = anchorFrames[0][sequence_length*(i):sequence_length*(i+1)]
                    tempAnchorFramesRNNInput = anchorFramesRNNInput
                    anchorFramesRemaining -= 16

                if positiveFramesRemaining < 16:
                    positiveFramesRNNInput = tempPositiveFramesRNNInput
                else:
                    positiveFramesRNNInput = positiveFrames[0][sequence_length*(i):sequence_length*(i+1)]
                    tempPositiveFramesRNNInput = positiveFramesRNNInput
                    positiveFramesRemaining -= 16

                if negativeFramesRemaining < 16:
                    negativeFramesRNNInput = tempNegativeFramesRNNInput
                else:
                    negativeFramesRNNInput = negativeFrames[0][sequence_length*(i):sequence_length*(i+1)]
                    tempNegativeFramesRNNInput = negativeFramesRNNInput
                    negativeFramesRemaining -= 16

                if i == quotient:
                    anchorFramesRNNInput = anchorFrames[0][actualFramesCount - sequence_length:actualFramesCount]
                    positiveFramesRNNInput = positiveFrames[0][positiveFramesCount - sequence_length:positiveFramesCount]
                    negativeFramesRNNInput = negativeFrames[0][negativeFramesCount - sequence_length:negativeFramesCount]

                if torch.cuda.is_available():
                    anchorFramesRNNInput = anchorFramesRNNInput.cuda()
                    positiveFramesRNNInput = positiveFramesRNNInput.cuda()
                    negativeFramesRNNInput = negativeFramesRNNInput.cuda()

                H, Hp, Hn = tripletRNNRGB(anchorFramesRNNInput.view(batch_size, sequence_length, input_size), positiveFramesRNNInput.view(batch_size, sequence_length, input_size), negativeFramesRNNInput.view(batch_size, sequence_length, input_size))
                print H, Hp, Hn

                loss = torch.zeros(1)
                if torch.cuda.is_available():
                    loss = loss.cuda()
                loss = Variable(loss)
                zero = torch.zeros(1)
                if torch.cuda.is_available():
                    zero = zero.cuda()
                zero = Variable(zero)
                loss += torch.sum(torch.max(zero, alpha - cos(H[0], Hp[0]) + cos(H[0], Hn[0])))
                loss /= sequence_length
                # print loss
                optimizerRGB.zero_grad()
                loss.backward()
                optimizerRGB.step()

torch.save(tripletRNNRGB.state_dict(), '/Users/prateek/8thSem/rl-person-verification/runs/model_run_rgb.pt')
torch.save(tripletRNNOP.state_dict(), '/Users/prateek/8thSem/rl-person-verification/runs/model_run_op.pt')
