import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
from torch.autograd import Variable

batch_size = 64

transforms.Compose([
    transforms.Resize(14),
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train():
    model.train()
    for epoch in range(1,2):
        for batch_idx, (image,label) in enumerate(train_loader):
            image, label = Variable(image), Variable(label)
            optimizer.zero_grad()
            output = model(image)
            loss = F.nll_loss(output, label)
            loss.backward()
            optimizer.step()

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for image, label in test_loader:
        image, label = Variable(image, volatile = True), Variable(label)
        output = model(image)
        # print output
        test_loss += F.nll_loss(output, label, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
train()
test()
