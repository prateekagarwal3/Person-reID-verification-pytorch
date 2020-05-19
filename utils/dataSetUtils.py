import os
import time
import torch
import buildModel
import prepareDataset
from PIL import Image
from torch.autograd import Variable

if sys.platform.startswith('linux'):
    dirPath = '/data/home/prateeka/'
elif sys.platform.startswith('darwin'):
    dirPath = '/Users/prateek/8thSem/'

seqRootRGB = dirPath + 'dataset/iLIDS-VID/i-LIDS-VID/sequences/'
seqRootOP = dirPath + 'dataset/iLIDS-VID-OF-HVP/sequences/'
personIdxDict, personFramesDict = prepareDataset.prepareDS(seqRootRGB)
personNoDict = dict([v,k] for k,v in personIdxDict.items())

featureDirRGB = dirPath + 'features/featuresVGG/RGB/'
featureDirOP = dirPath + 'features/featuresVGG/OP/'

for cam in range(1, 3):
    for i in range(1, 299, 3):
        k = 0
        pid1 = personIdxDict[i]
        pid2 = personIdxDict[i+1]
        pid3 = personIdxDict[i+2]
        # print personFramesDict[pid1][cam]
        fc1 = personFramesDict[pid1][cam-1]
        fc2 = personFramesDict[pid2][cam-1]
        fc3 = personFramesDict[pid3][cam-1]
        totalFramesCount = fc1 + fc2 + fc3

        totalFrames = torch.Tensor(totalFramesCount, 3, 224, 224)
        # print totalFramesCount
        camDir = "cam" + str(cam)
        fileName1 = os.path.join(seqRootOP, camDir, pid1)
        fileName2 = os.path.join(seqRootOP, camDir, pid2)
        fileName3 = os.path.join(seqRootOP, camDir, pid3)
        dirsList1 = sorted(os.listdir(fileName1))
        if dirsList1[0] == '.DS_Store':
            dirsList1.pop(0)
        dirsList2 = sorted(os.listdir(fileName2))
        if dirsList2[0] == '.DS_Store':
            dirsList2.pop(0)
        dirsList3 = sorted(os.listdir(fileName3))
        if dirsList3[0] == '.DS_Store':
            dirsList3.pop(0)
        for frame in dirsList1:
            frame = os.path.join(fileName1, frame)
            totalFrames[k] = prepareDataset.loadImage(frame)
            k += 1
        for frame in dirsList2:
            frame = os.path.join(fileName2, frame)
            totalFrames[k] = prepareDataset.loadImage(frame)
            k += 1
        for frame in dirsList3:
            frame = os.path.join(fileName3, frame)
            totalFrames[k] = prepareDataset.loadImage(frame)
            k += 1
        totalFrames = Variable(totalFrames)
        print totalFrames.size()
        tic = time.time()
        features = buildModel.cnn(totalFrames)
        # features = principalComponent(features, 64)
        print features.size()
        toc = time.time()
        print("Time taken by VGG :", str(toc-tic))
        print(fc1)
        print(fc2)
        print(fc3)
        fileName = os.path.join(featureDirOP, camDir, str(i))
        torch.save(features[0:fc1], fileName+".pt")

        fileName = os.path.join(featureDirOP, camDir, str(i+1))
        torch.save(features[fc1:fc1+fc2], fileName+".pt")

        fileName = os.path.join(featureDirOP, camDir, str(i+2))
        torch.save(features[fc1+fc2:fc1+fc2+fc3], fileName+".pt")
