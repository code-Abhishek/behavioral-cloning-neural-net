# **Behavioral Cloning using Keras**

## Writeup

The steps of this project were the following:
* Use a Udacity provided vehicle simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from camera images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the track without leaving the road

### Data Collection

Using the Udacity provided simulator, you can quickly generate large amounts of images/steering angle pairs to feed into the neural network. The network then creates a model that can be used to predict a future steering angle when presented with a new, never seen before image.

Here's a screenshot of the simulator:

![sim]

You can drive the car around the track manually, continually recording images from three cameras and the associated steering angle. The cameras record from the car's point of view, as if there were three different dashcams inside the car. Here is an example: 

#### Left camera
![left]

#### Center camera
![center]

#### Right camera
![right]

This is useful because it provides with network with triple the data compared to only having one camera. For the center image, you can simply feed in the image along with whatever the steering angle was at the time. For instance, in this example the car is headed into the side of the bridge, and the associated steering angle was 0.2 (this is a normalized number between -1 and 1, not actual degrees -- 1 being hard right, -1 being hard left). We can then use the left and right images with slightly adjusted steering angles.

This also provides the car with a large percentage of data that is not simply driving straight forward. If the entire dataset used to train the car was perfectly driven in the middle of the road, the car would have no way of learning to correct itself if it ever veered to the side. An effective and easy way to prevent this is to use these side camera angles to provide the car with a large amount of 'error' examples, essentially telling the car: "if you ever see yourself driving off to the side, this is how you correct yourself."

As I trained and tested the model, there were noticeable weak spots throughout the track. The car generally performed just fine on straight and even curved portions of the normal road, but there is a portion of the track towards the end where there is a dirt road turn off that the car would take:

#### Dirt turn off that confused the car
![dirt]

The rest of the track was relatively simple, getting the car to pass this section took a significant amount of my time. I tweaked the network architecture, added more training data, and changed the data augmentation process. Nothing seemed to work. It was finally overhauling the network architecture to follow NVIDIA's model (discussed later) that had a significant improvement.

### Model Architecture and Training Strategy

#### Data Augmentation and Preprocessing

As discussed previously, each datapoint recorded contained three images and a center steering measurement. Using an experimentally determined correction factor, each left and right image was assigned a steering measurement based on the center steering measurement. The left image's steering measurement was the center steering measurement plus the correction factor, and the right image's steering measurement was minus the correction factor.

Then, mirrored copies of all the images were created, effectively doubling the size of the dataset. For each mirrored image, the steering measurement was multiplied by -1.

Once the dataset was augmented, it was time to preprocess the data.

Normalization is always a good step when working with pixel data -- in order to have zero mean and equal variance, you can shift the pixel values from (0 to 255) to (-1 to 1). This helps the optimizer reduce the loss. Keras makes this easy with a Lambda function:

```
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
```

Next up: image cropping. The sky is a rather irrelevant part of the image, and does not help the car determine which way to turn. As such, I decided to crop it out. This can be done with Keras' `Cropping2D()` method:

```
model.add(Cropping2D(cropping=((70,25),(0,0))))
```

This removes the top 70 rows of pixels and the bottom 25 (which are mostly the car's hood).

I used a Python generator to lazily load the images into memory instead of all at once. My training dataset had about 8,000 data points, each containing three different images. This resulted in about 25,000 images, and after mirroring, was about 50,000. 50,000 images, 160x320 resolution with 3 channels is about 8GBs. Holding all of this in RAM is doable on some machines, but my AWS server kept killing the process. Thankfully, generators are great for this purpose. I wrote a `generator()` helper function that iterates through the list of CSV samples, loads the images and measurements into memory, mirrors the images, applies the correction factor to each left/right image, and returns a batch each time it's called.

`keras.fit_generator()` accepts a generator as an argument for both the training and the validation data, allowing me to easily split the dataset up and create a generator for the training and validation steps.

I had some minor issues creating the generator, and the Python debugger made it very easy to check variables and figure out what was going wrong. I only recently learned about the magical `import pdb`, and I've been using it constantly.

### Training the Neural Net

Using the magic of Keras, I was able to set up a very effective neural network in just a few lines of code. I originally started with the LeNet architecture, but switched over to the network used by NVIDIA in [their excellent paper](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64. After the convolution layers, I used one flatten layer and a series of fully connected layers, all narrowing down to one final output (the steering angle).

#### Attempts to reduce overfitting in the model

While training, I carefully observed the validation vs training loss to prevent overfitting if needed. However, the validation loss was always very similar to the training loss, indicating that the model was not overfitting. In previous implementations I experimented with dropout, but with my current network found it unneeded to achieve a successful lap.

#### Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

###  Final Model Architecture

As discussed previously, the model was heavily inspired by NVIDIA's paper.

Here is a visualization of the architecture:

![model]

### Conclusion

This was an excellent introduction to behavioral cloning -- a very cool technique to get surprisingly effective results without much effort. Data was not that difficult to generate, and I can see a significant amount of applications to other interesting problems.

[//]: # (Image References)

[left]: ./examples/left.jpg "Left Camera"
[center]: ./examples/center.jpg "Center Camera"
[right]: ./examples/right.jpg "Right Camera"
[sim]: ./examples/sim_screenshot.png "Screenshot of Simulator"
[dirt]: ./examples/dirt.png "Dirt turnoff"
[model]: ./examples/model.png "Neural Network"
