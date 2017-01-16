# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from imutils import paths
from keras.layers.convolutional import ZeroPadding2D,ZeroPadding1D
from keras.layers import Flatten, Dense, Dropout
import numpy as np
from keras.layers.convolutional import Convolution1D,Convolution2D, MaxPooling2D
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

for i in range(0,len(pedestriansLines), 1):
    pedestrianImage = cv2.imread(splitPath[0] + "/" + pedestriansLines[i].rstrip("\r\n"), cv2.IMREAD_COLOR )
    images.append(pedestrianImage.tolist())
    labels.append(1)

path = '/home/DataSets/CVC-Virtual-Pedestrian/train/background-frames/backgroundlist.txt'

backgroundFile = open(path)
backgroundLines = backgroundFile.readlines()

splitPath = path.rsplit('/', 1)

for j in range(0,len(backgroundLines), 1):
    backgroundImage = cv2.imread(splitPath[0] + "/" + backgroundLines[j].rstrip("\r\n"), cv2.IMREAD_COLOR)
    for z in range(0, 2, 1):
        xRand = random.randint(1, 500)
        yRand = random.randint(1, 300)
        xx = xRand + 48
        yy = yRand + 96
        croppedImage = backgroundImage[yRand:yy, xRand:xx]  # crops a fragment
        images.append(croppedImage.tolist())
        labels.append(0)

images = np.array(images)
labels = np_utils.to_categorical(labels, 2)
print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	images, labels, test_size=0.2)

# trainData = trainData.reshape(len(trainData), 96, 48,1).astype('float32')
# testData = testData.reshape(len(testData), 96, 48,1).astype('float32')
#trainData /= 255
#testData /= 255
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(96,48,3)))
model.add(Convolution2D(86, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(66, 3, 3, activation='relu'))
model.add(MaxPooling2D((3,3), strides=(3,3)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(44, 3, 3, activation='relu'))
model.add(MaxPooling2D((3,3), strides=(3,3)))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(22, 3, 3, activation='relu'))
model.add(MaxPooling2D((3,3), strides=(3,3)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(15))
model.add(Dense(2))
model.add(Activation("softmax"))

print("[INFO] compiling model...")
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss="binary_crossentropy", optimizer=adam,
	metrics=["accuracy"])

model.fit(trainData, trainLabels, nb_epoch=20,batch_size=64,
	verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=64, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
