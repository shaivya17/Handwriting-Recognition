from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import *


class Sample:

	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:

	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:

	def __init__(self, filePath, batchSize, imgSize, maxTextLen):


		assert filePath[-1]=='/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []

		f=open(filePath+'words.txt')
		chars = set()
		bad_samples = []
		bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
		for line in f:
			# ignore comment line
			if not line or line[0]=='#':
				continue

			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9

			# filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
			gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
			chars = chars.union(set(list(gtText)))


			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue


			self.samples.append(Sample(gtText, fileName))

		if set(bad_samples) != set(bad_samples_reference):
			print("Warning, damaged images found:", bad_samples)
			print("Damaged images expected:", bad_samples_reference)

		# split into training and validation set: 95% - 5%
		splitIdx = int(0.95 * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		# put words into lists
		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]


		self.numTrainSamplesPerEpoch = 25000

		self.trainSet()

		# list of all chars in dataset
		self.charList = sorted(list(chars))


	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text


	def trainSet(self):
		"switch to randomly chosen subset of training set"
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]


	def validationSet(self):
		"switch to validation set"
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples


	def getIteratorInfo(self):
		"current batch index and overall number of batches"
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


	def hasNext(self):
		"iterator"
		return self.currIdx + self.batchSize <= len(self.samples)


	def getNext(self):
		"iterator"
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, imgs)


