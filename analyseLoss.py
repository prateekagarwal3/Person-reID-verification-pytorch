import torch
import matplotlib.pyplot as plt

f = open('loss90pdata20epochs.txt', 'r')
Y = []
y = []

for line in f.readlines():
    Y.append(float(line))

for x in Y:
    if x > 1:
        y.append(x)

for a in y:
    Y.remove(a)

X = [x for x in range(len(Y))]

plt.plot(X, Y)
plt.show()


'''
features = torch.load('/Users/prateek/8thSem/temporalRepresentation/cam1/2.pt')
features = features[:, 0:64]
features = torch.mean(features, dim=0).view(64, 1)

features1 = torch.load('/Users/prateek/8thSem/temporalRepresentation/cam2/2.pt')
features1 = features1[:, 0:64]
features1 = torch.mean(features1, dim=0).view(64, 1)

fig,ax = plt.subplots(1)
ax.plot(features.numpy())
ax.plot(features1.numpy())

plt.show()'''
