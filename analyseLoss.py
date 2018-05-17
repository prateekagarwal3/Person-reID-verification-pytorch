import torch
import torch.nn as nn
import matplotlib.pyplot as plt

f = open('lossRGBv.txt', 'r')
Y = []
y = []
for line in f.readlines():
    Y.append(float(line))

plt.plot(Y)
plt.show()


# features = torch.load('/Users/prateek/8thSem/temporalRepresentation/cam1/1.pt')
# features = features[:, 0:64]
# # # features = torch.mean(features, dim=0).view(64, 1)
# #
# # features1 = torch.load('/Users/prateek/8thSem/temporalRepresentation/cam1/2.pt')
# # features1 = features1[:, 0:64]
# # features1 = torch.mean(features1, dim=0).view(64, 1)
#
# fig,ax = plt.subplots(1)
# ax.plot(features[0].numpy())
# ax.plot(features[1].numpy())
# ax.plot(features[2].numpy())
# ax.plot(features[3].numpy())
# ax.plot(features[4].numpy())
#
# plt.show()
