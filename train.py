import csv
import cv2
import numpy as np

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
    measurements.append(float(line[3]))
    
X_train = np.array(images) 
y_train = np.array(measurements) 

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('my_first_model.h5')


