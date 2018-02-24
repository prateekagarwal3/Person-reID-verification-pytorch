import prepareDataset
import tripletNetwork

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
num_epochs = 2
batch_size = 1
testTrainSplit = 0.8

seqRootRGB = '/Users/prateek/8thSem/rl-person-verification/dataset/iLIDS-VID/i-LIDS-VID/sequences/'
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
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        out, hidden_data = self.lstm(x, (h0, c0))
        return out

rnn = RNN(input_size, hidden_size, num_layers)
tripletRNN = tripletNetwork.TripletNet(rnn)
if torch.cuda.is_available():
    tripletRNN().cuda()

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
optimizer = torch.optim.SGD(tripletRNN.parameters(), lr=learning_rate, momentum=momentum)

torch.backends.cudnn.benchmark = True
count = 1
tripletRNN.train()
for epoch in range(num_epochs):
    print("Epoch Loop Running", epoch)
    for triplet in trainTriplets:
        triplet = [15,15,213]
        print("Triplet being used : ", triplet)
        count += 1
        if(count > 2):
            break
        for cam in range(1,2):
            #print("Cam Loop Running")
            anchor = personIdxDict[triplet[0]]
            positive = personIdxDict[triplet[1]]
            negative = personIdxDict[triplet[2]]

            frameCountList = personFramesDict[anchor]
            anchorFrameCount = frameCountList[cam-1]
            frameCountList = personFramesDict[positive]
            positiveFrameCount = frameCountList[2-cam]
            frameCountList = personFramesDict[negative]
            negativeFrameCount = frameCountList[2-cam]
            actualFrameCount = max(anchorFrameCount, positiveFrameCount, negativeFrameCount)
            # print(anchorFrameCount, positiveFrameCount, negativeFrameCount, actualFrameCount)

            anchorFrames = prepareDataset.getPersonFrames(seqRootRGB, anchor, cam, anchorFrameCount)
            positiveFrames = prepareDataset.getPersonFrames(seqRootRGB, positive, 3-cam, positiveFrameCount)
            negativeFrames = prepareDataset.getPersonFrames(seqRootRGB, negative, 3-cam, negativeFrameCount)

            quotient = actualFrameCount / sequence_length
            remainder = actualFrameCount % sequence_length

            anchorFramesRemaining = anchorFrameCount
            positiveFramesRemaining = positiveFrameCount
            negativeFramesRemaining = negativeFrameCount

            tempAnchorFramesRNNInput = anchorFrames[0:16]
            tempPositiveFramesRNNInput = positiveFrames[0:16]
            tempNegativeFramesRNNInput = negativeFrames[0:16]

            # print("Q, R:", quotient, remainder)
            for i in range(quotient):
                # print("Quotient Loop Running")

                if anchorFrameRemaining < 0:
                    anchorFramesRNNInput = tempAnchorFramesRNNInput
                else:
                    anchorFramesRNNInput = anchorFrames[sequence_length*(i):sequence_length*(i+1)]
                    tempAnchorFramesRNNInput = anchorFramesRNNInput
                    anchorFrameRemaining -= 16

                if positiveFrameRemaining < 0:
                    positiveFramesRNNInput = tempPositiveFramesRNNInput
                else:
                    positiveFramesRNNInput = positiveFrames[sequence_length*(i):sequence_length*(i+1)]
                    tempPositiveFramesRNNInput = positiveFramesRNNInput
                    positiveFrameRemaining -= 16

                if negativeFrameRemaining < 0:
                    negativeFramesRNNInput = tempNegativeFramesRNNInput
                else:
                    negativeFramesRNNInput = negativeFrames[sequence_length*(i):sequence_length*(i+1)]
                    tempNegativeFramesRNNInput = negativeFramesRNNInput
                    negativeFrameRemaining -= 16

                H, Hp, Hn = tripletRNN(anchorFramesRNNInput, positiveFramesRNNInput, negativeFramesRNNInput)
                loss = Variable(torch.zeros(1))
                # print H, Hp, Hn
                for j in range(sequence_length):
                    loss = loss + torch.max(Variable(torch.zeros(1)), Variable(alpha) - cos(H[j], Hp[j]) + cos(H[j], Hn[j]))
                loss = loss / sequence_length
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            H, Hp, Hn = tripletRNN(anchorFrames[anchorFrameCount - sequence_length:actualFrameCount], positiveFrames[positiveFrameCount - sequence_length:actualFrameCount], negativeFrames[negativeFrameCount - sequence_length:actualFrameCount])
            loss = Variable(torch.zeros(1))
            for j in range(sequence_length):
                loss = loss + torch.max(Variable(torch.zeros(1)), Variable(alpha) - cos(H[j], Hp[j]) + cos(H[j], Hn[j]))
            loss = loss / sequence_length
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
