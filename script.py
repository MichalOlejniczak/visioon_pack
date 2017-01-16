# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
from skimage.feature import hog
import cv2
import random
import os

orientation = 9
pixelsPerCell = (6, 6)
cellsPerBlock = (3, 3)

path = '/home/DataSets/CVC04-Virtual2/Crops_48x96/files.txt'

pedestriansFile = open(path)
pedestriansLines = pedestriansFile.readlines()

images = []
labels = []
computedHog = []
splitPath = path.rsplit('/', 1)

for i in range(0, len(pedestriansLines), 1):
    pedestrianImage = cv2.imread(splitPath[0] + "/" + pedestriansLines[i].rstrip("\r\n"), 0)
    images.append(pedestrianImage)
    labels.append(1)

path = '/home/DataSets/CVC-Virtual-Pedestrian/train/background-frames/backgroundlist.txt'

backgroundFile = open(path)
backgroundLines = backgroundFile.readlines()

splitPath = path.rsplit('/', 1)

for j in range(0, len(backgroundLines), 1):
    backgroundImage = cv2.imread(splitPath[0] + "/" + backgroundLines[j].rstrip("\r\n"), 0)
    for z in range(0, 10, 1):
        xRand = random.randint(1, 500)
        yRand = random.randint(1, 300)
        xx = xRand + 48
        yy = yRand + 96
        croppedImage = backgroundImage[yRand:yy, xRand:xx]  # crops a fragment
        images.append(croppedImage)
        labels.append(0)



for i in range(0, len(images), 1):
    currentHog = hog(images[i], orientations=orientation, pixels_per_cell=pixelsPerCell,
                     cells_per_block=cellsPerBlock, transform_sqrt=True, feature_vector=True)
    #computedHog.append(np.array([2,2,2,2,2]))
    #computedHog.append([2,2,2,2,2])
    computedHog.append(currentHog.tolist())
computedHog = np.array(computedHog)
print()
print()
print(computedHog[0])
print(len(labels))
print(len(computedHog))
print()
print()
labels = np_utils.to_categorical(labels, 2)
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	computedHog, labels, test_size=0.25)

model = Sequential()
model.add(Dense(768, input_dim=6804, init="uniform",
	activation="relu"))
model.add(Dense(384, init="uniform", activation="relu"))
model.add(Dense(2))
model.add(Activation("softmax"))

print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50,batch_size=128,
	verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
