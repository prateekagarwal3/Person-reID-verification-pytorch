import buildModel
import prepareDataset

import os
import time
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

vgg = models.alexnet(pretrained=True)
vgg = vgg.cuda()
def img_load(file_name):
	img = Image.open(file_name)
	image_transform = transforms.Compose([
		# transforms.Pad((80, 48), fill=0),
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	img = image_transform(img)
	img = img.unsqueeze_(0)
	return img.cuda()

img = img_load('/data/home/prateeka/dataset/iLIDS-VID/i-LIDS-VID/images/cam1/person001/cam1_person001.png')
tic = time.time()
print vgg(img)
toc = time.time()
print(str(toc-tic) + 's')
