import csv
import cv2
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import sklearn
import random
from pathlib import Path
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

correction = 0.25 
batch_size = 32

def path_to_filename(path):
    """ Pulls the filename out of a path by splitting on the slashes
    """
    if 'C:' in path:
        return path.split('\\')[-1] # Grab the Windows file name, cutting off the base path
    else:
        return path.split('/')[-1] # Grab the file name, cutting off the base path

def generator(samples, batch_size=32):
    """Takes the a list of lines from a CSV and generates a list of images
    and steering measurements
    """
    n_samples = len(samples)
    while True: # Loop forever so the generator never exits, as expected by keras.fit_generator
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for sample in batch_samples: # This is a line of the CSV file
                for i in range(3):
                    source_path = sample[i] # line[0],[1],[2] are the center/left/right pics  
                    filename = path_to_filename(source_path)
                    path = './data/IMG/' + filename
                    image = cv2.imread(path)
                    if image is None:
                        # Did you know that imread quietly returns None if it can't find the image? Kind of unhelpful. 
                        print('Error! Cannot find image')
                        pdb.set_trace()
                    images.append(image)
                # Append three measurements to the list
                measurement = float(sample[3])
                measurements.append(measurement) # center camera 
                measurements.append(measurement + correction) # left camera 
                measurements.append(measurement - correction) # right camera

            # After we've run through the entire batch, go through and augment the data via mirroring
            aug_images = []
            aug_measurements = []
            for image, measurement in zip(images, measurements):
                flipped_image = cv2.flip(image, 1)
                flipped_measurement = measurement * -1.0
                aug_images.append(flipped_image)
                aug_measurements.append(flipped_measurement)
            images += aug_images
            measurements += aug_measurements

            X_train = np.array(images) 
            y_train = np.array(measurements) 
            # yield (X_train, y_train) # inputs, targets
            yield sklearn.utils.shuffle(X_train, y_train) # inputs, targets

# Read all CSVs that I'm using 
PATH = Path('data/CSVs')
CSVs = list(PATH.iterdir())

lines = []
for CSV in CSVs:
    with open(str(CSV)) as file: 
        reader = csv.reader(file)
        tmp_lines = []
        for line in reader:
            tmp_lines.append(line)
        print('Reading file', str(CSV), 'it has', len(tmp_lines), 'lines')
        tmp_lines = tmp_lines[1:] # discard first line as it is just titles, not data
        lines += tmp_lines

# pdb.set_trace()
random.shuffle(lines) # Shuffle the order
print('Using', len(lines), 'different data points with three images each')

train, val = train_test_split(lines, test_size=0.2) # Split the CSV into a test/val dataset

train_gen = generator(train, batch_size)
val_gen = generator(val, batch_size)

model = Sequential()
# Normalize and crop the images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=2, activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen, steps_per_epoch = len(train) / batch_size, \
    epochs=5, validation_data=val_gen, validation_steps = len(val) / batch_size) # Shuffle is on by default


print('Saving model to file...')
model.save('steering.h5')
