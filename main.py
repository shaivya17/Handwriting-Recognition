from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import *
from Model import *
from SamplePreprocessor import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PyQt5.QtGui import QImage, QPalette, QBrush


class mywin(QMainWindow):
	def __init__(self):
		super(mywin, self).__init__()
		self.setWindowTitle("Hand Writing Recognition")
		self.setGeometry(QRect(600, 300, 400, 500))
		oImage = QImage('../data/test.png')
		sImage = oImage.scaled(QSize(300, 200))
		palette = QPalette()
		palette.setBrush(10, QBrush(sImage))
		self.setPalette(palette)

		self.setCentralWidget(QWidget(self))
		self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowMinimizeButtonHint);
		self.addMenus()
		self.w = self.h = 512
		self.winCount = 0
	def addMenus(self):
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		openMenu = QMenu('Open', self)
		impAct = QAction('Open image', self)
		impAct.triggered.connect(main)
		openMenu.addAction(impAct)
		exmenu = QMenu('Exit', self)
		exact = QAction('Close File', self)
		exact.triggered.connect(self.closeApp)
		exmenu.addAction(exact)
		fileMenu.addMenu(openMenu)
		fileMenu.addMenu(exmenu)
		self.show()

	def closeApp(self):
		quit()


def fileopen():
	dlg = QFileDialog()
	if (dlg.exec()):
		openfilepath = dlg.selectedFiles()
		return (openfilepath[0])


class FilePaths:
    fnCharList = '../model/charList.txt'
    fnAccuracy = '../model/accuracy.txt'
    fnTrain = '../data/'
    fnCorpus = '../data/corpus.txt'


def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 5  # stop training after this number of epochs without improvement
    while True:
        epoch += 1


        # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)


        # validate
        charErrorRate = validate(model, loader)

        # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:

            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:

            noImprovementSince += 1

        # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:

            break


def validate(model, loader):
    "validate NN"

    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0


    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()

        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)


        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])

    # print validation result
    charErrorRate = numCharErr / numCharTotal


    return charErrorRate


def infer(model, fnImg):
    "recognize text in image provided by file path"
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)
    print('Recognized:', '"' + recognized[0] + '"')
    print('Probability:', probability[0])
    msgBox = QMessageBox()
    msgBox.setText("The word is  " + recognized[0])
    msgBox.exec()


def main():

    decoderType = DecoderType.BestPath
    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
    infer(model, fileopen())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = mywin()
    ex.show()
    sys.exit(app.exec())
