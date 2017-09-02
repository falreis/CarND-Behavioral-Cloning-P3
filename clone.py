import csv
import cv2
import numpy as np
import gc

LOG_PATH = "data/small/"

#parameters to tune
correction = 0.1
cropping_top = 75
cropping_bot = 30
cropping_lef = 5
cropping_rig = 5

#read file

images = []
mesurements = []

with open(LOG_PATH + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        image_c = cv2.imread(LOG_PATH + line[0])
        mesurement_c = float(line[3])

        image_l = cv2.imread(LOG_PATH + line[1])
        image_r = cv2.imread(LOG_PATH + line[2])
        mesurement_l = mesurement_c + correction
        mesurement_r = mesurement_c - correction

        images.append(image_c)
        images.append(image_l)
        images.append(image_r)
        mesurements.append(mesurement_c)
        mesurements.append(mesurement_l)
        mesurements.append(mesurement_r)

        #images.append(cv2.flip(image_c, 1))
        #mesurements.append(-mesurement_c)
#endfor

x_train = np.array(images)
y_train = np.array(mesurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((cropping_top, cropping_bot),(0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

#model.add(Flatten())
#model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(x_train, y_train, validataion_split=0.2, huffle=True, nb_epoch=1)
model.fit(x_train, y_train, nb_epoch=2, validation_split=0.2, shuffle=True)

#run garbage collector
gc.collect()

#save model
model.save('model.h5')

print("THE END")

#loss: 0.1356 - val_loss: 0.0807 (10 epochs - 1 layer)
#loss: 24.7414 - val_loss: 7.1673 (3 epochs - LeNet)