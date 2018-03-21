import csv
import cv2
import numpy as np
import pdb
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

correction = 0.5 # degrees? Probably too small
batch_size = 32

def path_to_filename(path):
    """ Pulls the filename out of a path by splitting on the slashes
    """
    if 'C:' in path:
        return path.split('\\')[-1] # Grab the Windows file name, cutting off the base path
    else:
        return path.split('/')[-1] # Grab the file name, cutting off the base path

def generator(samples, batch_size=32):
    """Takes the a list of lines from a CSV and generates a list of image filenames
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
                    source_path = line[i] # line[0],[1],[2] are the center/left/right pics  
                    filename = path_to_filename(source_path)
                    path = './data/IMG/' + filename
                    image = cv2.imread(path)
                    if image is None:
                        # Did you know that imread quietly returns None if it can't find the image? Kind of unhelpful. 
                        print('Error! Cannot find image')
                        pdb.set_trace()
                    images.append(image)
                # Append three measurements to the list
                measurement = float(line[3])
                measurements.append(measurement) # center camera 
                measurements.append(measurement + correction) # left camera 
                measurements.append(measurement - correction) # right camera

            # Add mirror image data to the dataset
            # for image, measurement in zip(images, measurements):
               # flipped_image = cv2.flip(image, 1)
               # flipped_measurement = float(measurement) * -1.0
               # images.append(flipped_image)
               # measurements.append(flipped_measurement)

            X_train = np.array(images) 
            y_train = np.array(measurements) 
            yield (X_train, y_train) # inputs, targets
            # yield sklearn.utils.shuffle(X_train, y_train) # inputs, targets

# Read the CSV from Udacity
lines = []
tmp_lines = []
with open('data/driving_log.csv') as file: 
    reader = csv.reader(file)
    for line in reader:
        lines.append(line)

lines = lines[1:] # discard first line as it is just titles, not data

# Add custom dataset 
with open('sdc-sim/driving_log.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        tmp_lines.append(line)

tmp_lines = tmp_lines[1:]
lines += tmp_lines

train, val = train_test_split(lines, test_size=0.2) # Split the CSV into a test/val dataset

train_gen = generator(train, batch_size)
val_gen = generator(val, batch_size)
pdb.set_trace()

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen, steps_per_epoch = len(train) / batch_size, \
    epochs=5, validation_data=val_gen, validation_steps = len(val) / batch_size, verbose=2) # Shuffle is on by default


print('Saving model to file...')
model.save('steering.h5')

