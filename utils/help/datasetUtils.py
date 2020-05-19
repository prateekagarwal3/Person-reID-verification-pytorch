import torch

# given the dataset, which consists of a table where t[x] contains the images for person x
# split the dataset into testing and training parts
def partitionDataset(nTotalPersons,testTrainSplit):
	splitPoint = int(nTotalPersons * testTrainSplit)
	inds = torch.randperm(nTotalPersons)

	# -- save the inds to a mat file
	# --mattorch.save('rnnInds.mat',inds)

	trainInds = inds[0:splitPoint]
	testInds = inds[(splitPoint):nTotalPersons+1]

	# -- save the split to a file for later use
	# -- datasetSplit = {
	# --     trainInds = trainInds,
	# --     testInds = testInds,
	# -- }
	# -- torch.save('./trainedNets/dataSplit_PRID2011.th7',datasetSplit)
	return trainInds,testInds

	# -- the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
	# -- choose a pair of sequences from the same person
def getPosSample(dataset, trainInds, person, sampleSeqLen):

	# -- choose the camera, ilids video only has two, but change this for other datasets
	camA = 1
	camB = 2

	actualSampleSeqLen = sampleSeqLen
	nSeqA = dataset[trainInds[person]][camA].size(0)
	nSeqB = dataset[trainInds[person]][camB].size(0)

	# -- what to do if the sequence is shorter than the sampleSeqLen
	if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
		if nSeqA < nSeqB:
			actualSampleSeqLen = nSeqA
		else:
			actualSampleSeqLen = nSeqB

	startA = torch.floor(torch.rand(1)[1] * ((nSeqA - actualSampleSeqLen) + 1)) + 1
	startB = torch.floor(torch.rand(1)[1] * ((nSeqB - actualSampleSeqLen) + 1)) + 1

	return startA,startB,actualSampleSeqLen

# -- the dataset format is dataset[person][camera][nSeq][nCrop][FeatureVec]
# -- choose a pair of sequences from different people
def getNegSample(dataset,trainInds,sampleSeqLen):
	permAllPersons = torch.randperm(trainInds.size(0))
	personA = permAllPersons[1]--torch.floor(torch.rand(1)[1] * 2) + 1
	personB = permAllPersons[2]--torch.floor(torch.rand(1)[1] * 2) + 1

	# -- choose the camera, ilids video only has two, but change this for other datasets
	camA = torch.floor(torch.rand(1)[1] * 2) + 1
	camB = torch.floor(torch.rand(1)[1] * 2) + 1

	actualSampleSeqLen = sampleSeqLen
	nSeqA = dataset[trainInds[personA]][camA].size(0)
	nSeqB = dataset[trainInds[personB]][camB].size(0)

	# -- what to do if the sequence is shorter than the sampleSeqLen
	if nSeqA <= sampleSeqLen or nSeqB <= sampleSeqLen:
		if nSeqA < nSeqB:
			actualSampleSeqLen = nSeqA
		else:
			actualSampleSeqLen = nSeqBs

	startA = torch.floor(torch.rand(1)[1] * ((nSeqA - actualSampleSeqLen) + 1)) + 1
	startB = torch.floor(torch.rand(1)[1] * ((nSeqB - actualSampleSeqLen) + 1)) + 1

	return personA,personB,camA,camB,startA,startB,actualSampleSeqLen
