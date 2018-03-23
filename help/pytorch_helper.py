import time
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torchvision import transforms, datasets
from torch.autograd import Variable
from sklearn.decomposition import PCA

def rgb2yuv():
    img = Image.open('test.jpeg')
    img_yuv = img.convert('YCbCr')
    return img_yuv

def torchToNumpy():
    tch = torch.ones(5,10)
    print tch
    Np = tch.numpy()
    print Np
    tch = torch.from_numpy(Np)
    print tch

def scanDirectory(rootdir):
    print sorted(os.listdir(rootdir))
    for subdir, dirs, files in os.walk(rootdir):
        print files

def imagePadding():
    img = Image.open('/Users/prateek/8thSem/rl-person-verification/dataset/PRID2011/multi_shot/cam_a/person_0001/0125.png')
    img = transforms.Pad((80, 48), fill=0)(img)
    img = transforms.ToTensor()(img)
    return Variable(img.unsqueeze_(0))

def preTrainedModel(img):
    vgg = models.vgg16_bn(pretrained=True)
    mod = list(vgg.classifier.children())
    new_classifier = torch.nn.Sequential(*mod[:1])
    vgg.classifier = new_classifier
    return vgg(img)

def emptyTensor():
    return torch.Tensor(10, 3, 224, 224)

def randomSplit():
    nTotalPersons = 300
    testTrainSplit = 0.5
    splitPoint = int(nTotalPersons * testTrainSplit)
    inds = torch.randperm(nTotalPersons)
    trainInds = inds[0:splitPoint]
    testInds = inds[(splitPoint):nTotalPersons+1]

def transposeList(l):
    return np.array(l).T.tolist()

def principal_component(X):
    X = X.data
    X = X.numpy()
    X = X.T
    # X = np.random.random((250,4096))
    pca = PCA(n_components=64)
    # print X.shape
    Y = pca.fit_transform(X)
    # print reduced.shape
    return torch.from_numpy(Y.T)


def getSize():
    x = torch.zeros(2,3,4)
    print x.size(2)

def opticalFlowSizeCheck():
    img = Image.open('/Users/prateek/8thSem/rl-person-verification/dataset/iLIDS-VID-OF-HVP/sequences/cam1/person001/cam1_person001_00317.png')
    print img
    img = transforms.ToTensor()(img)
    print img

def tensorStack():
    a = torch.ones(10,4096)
    b = torch.zeros(10,4096)
    f = torch.ones(10,4096)
    g = torch.zeros(10,4096)
    varList = [[a, b], [f, g]]
    # e = torch.stack(a, dim=0, out=a)
    varList[0] = torch.cat(varList[0], dim=0)
    varList[1] = torch.cat(varList[1], dim=0)
    # print e
    print varList

def runTime():
    startTime = time.time()
    print startTime
    x = 0
    for i in range(10000000):
        x = x+1
    endTime = time.time()
    print(endTime-startTime)

def dimensionAdder():
    X = torch.ones(23,128)
    X.unsqueeze_(1)
    print X

def modelParameters():
    model = models.alexnet(pretrained=False)
    for name, param in model.state_dict().iteritems():
        print name, param.size()

principal_component()
