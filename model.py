import csv
import cv2
import numpy as np
import gc
import random
import sklearn

#LOG_PATHS = ["data/2laps/", "data/2laps-reverse/", "data/bridge/", "data/reckless/"]
#LOG_PATHS = ["data/2laps/", "data/2laps-reverse/", "data/bridge/"]
LOG_PATHS = ["data/bridge/"]

#parameters to tune
correction = 0.22
crop_top = 70
crop_bottom = 25
crop_left = 0
crop_right = 0
stability = 1.1
dropout_rate = 0.3

#read file
images = []
mesurements = []

#generator
'''
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32
'''

#read file
for path in LOG_PATHS:
    with open(path + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            image_c = cv2.imread(path + line[0])
            image_l = cv2.imread(path + line[1])
            image_r = cv2.imread(path + line[2])
            
            #cv2.normalize(image_c, image_c, 0, 255, cv2.NORM_L1)
            #cv2.normalize(image_l, image_c, 0, 255, cv2.NORM_L1)
            #cv2.normalize(image_r, image_c, 0, 255, cv2.NORM_L1)

            #mesurements
            mesurement_c = float(line[3])
            if mesurement_c > 0:
                mesurement_c = mesurement_c * stability
            else:
                mesurement_c = mesurement_c / stability

            images.append(image_c)
            mesurements.append(mesurement_c)
                
            mesurement_l = mesurement_c + correction
            images.append(image_l)
            mesurements.append(mesurement_l)
                
            mesurement_r = mesurement_c - correction
            images.append(image_r)
            mesurements.append(mesurement_r)

            #flip images
            #images.append(np.fliplr(image_c))
            #images.append(np.fliplr(image_l))
            #images.append(np.fliplr(image_r))

            #mesurements.append(-mesurement_c)
            #mesurements.append(-mesurement_l)
            #mesurements.append(-mesurement_r)
        #endfor
    #endwith
#endfor

x_train = np.array(images)
y_train = np.array(mesurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((crop_top, crop_bottom),(crop_left, crop_right))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(dropout_rate))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dropout(dropout_rate))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(dropout_rate))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, nb_epoch=2, validation_split=0.2, shuffle=True)
#model.fit_generator(x_train, samples_per_epoch= len(x_train), /
#            validation_data=y_train, nb_val_samples=len(y_train), nb_epoch=2)

#run garbage collector
gc.collect()

#save model
model.save('model.h5')

print("THE END")
