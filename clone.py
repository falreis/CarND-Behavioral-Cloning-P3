import csv
import cv2
import numpy as np
import gc

LOG_PATH = "data/small/"

driving_log = []
center_paths = []
left_paths = []
right_paths = []

images = []
mesurements = []

with open(LOG_PATH + "driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        driving_log.append(line)

for path in driving_log:
    center_paths.append(path[0])
    left_paths.append(path[1])
    right_paths.append(path[2])

    images.append(cv2.imread(LOG_PATH + path[0]))
    mesurements.append(float(path[3]))
#endfor

x_train = np.array(images)
y_train = np.array(mesurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda 
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
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
model.fit(x_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)

#run garbage collector
gc.collect()

#save model
model.save('model.h5')

print("THE END")

#loss: 0.1356 - val_loss: 0.0807 (10 epochs - 1 layer)
#loss: 24.7414 - val_loss: 7.1673 (3 epochs - LeNet)