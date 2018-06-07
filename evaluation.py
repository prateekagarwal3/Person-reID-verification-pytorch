import torch
import random
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score

all = [x for x in range(1, 301)]
random.shuffle(all)
all = np.asarray(all)
queryid = all[0:150]
trainingid = all[151:300]

galleryid = queryid

gallery = len(galleryid)
query = len(queryid)

similarity_matrix = torch.Tensor(query, gallery)
features_dir = '/Users/prateek/8thSem/ilids_features/'
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

for i in range(query):
    for j in range(gallery):
        features_id1 = torch.load(features_dir + 'cam1/' + str(queryid[i]) + '.pt', map_location='cpu')
        features_id2 = torch.load(features_dir + 'cam2/' + str(galleryid[j]) + '.pt', map_location='cpu')
        similarity_matrix[i, j] = cos(torch.mean(features_id1, dim=0), torch.mean(features_id2, dim=0)).item()

print similarity_matrix

cmc = np.zeros(gallery, dtype=np.float32)

# a = np.arange(query
# b = np.arange(gallery)

mask = galleryid[None] == queryid[:,None]

apArr = []
for i in range(query):
    k = np.where(mask[i,np.flip(np.argsort(similarity_matrix[i]),axis = 0) ])[0][0]
    cmc[k:] +=1
    ap = average_precision_score(mask[i], similarity_matrix[i])
    apArr.append(ap)

apArr = np.asarray(apArr)
mAp = np.mean(apArr)

print cmc
cmc = cmc / query

print('mAP: {:.2%} | top-1: {:.2%} top-5: {:.2%} | top-10: {:.2%} | top-20: {:.2%}'.format(
        mAp, cmc[0], cmc[4], cmc[9], cmc[19]))
