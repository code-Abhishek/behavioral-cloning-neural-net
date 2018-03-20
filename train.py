import csv
import cv2
import numpy as np
import pdb
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read the CSV from Udacity
lines = []
with open('data/driving_log.csv') as file: 
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

# Add custom dataset 
with open('sdc-sim/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
lines = lines[1:] # discard first line as it is just titles, not data
correction = 0.5 # degrees? Probably too small
for line in lines:
    # Append three images to the list
    for i in range(3):
        source_path = line[i] # line[0],[1],[2] are the center/left/right pics  
        filename = source_path.split('/')[-1] # Grab the file name, cutting off the base path
        path = './data/IMG/' + filename
        image = cv2.imread(path)
        images.append(image)
    # Append three measurements to the list
    measurement = float(line[3])
    measurements.append(measurement) # center camera 
    measurements.append(measurement + correction) # left camera 
    measurements.append(measurement - correction) # right camera

X_train = np.array(images) 
y_train = np.array(measurements) 

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

# Augment the data and fit again due to memory constraints -- you can't do it all at once
aug_images = []
aug_measurements = []
for image, measurement in zip(images, measurements):
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    aug_images.append(flipped_image)
    aug_measurements.append(flipped_measurement)

X_train = np.array(aug_images) 
y_train = np.array(aug_measurements) 

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

print('Saving model to file...')
model.save('steering.h5')




