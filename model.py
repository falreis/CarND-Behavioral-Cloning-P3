import csv
import cv2
import numpy as np
import gc
import random
import sklearn

LOG_PATHS = ["data/2laps/", "data/2laps-reverse/", "data/bridge/", "data/reckless/"]

#parameters to tune
correction = 0.22
crop_top = 70
crop_bottom = 25
crop_left = 0
crop_right = 0
stability = 1.1
dropout_rate = 0.3

#read file
samples = []
for path in LOG_PATHS:
    with open(path + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            for i in range(0,3):
                line[i] = path + line[i]
            samples.append(line)
        #endfor
    #endwith
#endfor

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: #for i in (0,1000): # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #images
                center_image = cv2.imread(batch_sample[0])
                left_image = cv2.imread(batch_sample[1])
                right_image = cv2.imread(batch_sample[2])

                #angles
                center_angle = float(batch_sample[3])
                if center_angle > 0:
                    center_angle *= stability
                else:
                    center_angle /= stability

                #append images and angles
                images.append(center_image)
                images.append(left_image) #left image
                images.append(right_image) #right image
                angles.append(center_angle)
                angles.append(center_angle + correction) #left angle
                angles.append(center_angle - correction) #right angle

                #flipped images and angles
#                images.append(np.fliplr(center_image))
#                images.append(np.fliplr(left_image))
#                images.append(np.fliplr(right_image))
#                angles.append(-center_angle)
#                angles.append(-(center_angle + correction))
#                angles.append(-(center_angle - correction))
            #endfor

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        #endfor
    #endwhile
#enddef generator

#train and validation generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#neural network architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x / 255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((crop_top, crop_bottom),(crop_left, crop_right))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(dropout_rate))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Dropout(dropout_rate))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout(dropout_rate))
model.add(Dense(10))
model.add(Dense(1))

#NN compile and fit generator
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), 
    validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=1)

#run garbage collector
gc.collect()

#save model
model.save('model.h5')

print("THE END")
