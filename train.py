import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('data/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
lines = lines[1:] # discard first line as it is just titles, not data
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1] # Grab the file name, cutting off the base path
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    images.append(flipped_image)
    measurements.append(flipped_measurement)
    
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

model.save('steering.h5')

