
# coding: utf-8

# # Behaviorial Cloning Project
# 
# ## Udacity - Self-Driving Car NanoDegree Term-1
# 
# ### Overview
# 
# In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.
# We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.
# We also want you to create a detailed writeup of the project. Check out the writeup template for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

# In[1]:


# Imports for the Project

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Activation, Dropout

import csv

import cv2
from skimage import exposure

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random


# In[2]:


############################### Command Center of the Network ###############################

# The file containing data collection to be used for training
training_sets = []

training_sets.append('./data/driving_log.csv')
training_sets.append('./Tight Turns/driving_log.csv')
training_sets.append('./Off Track Data/driving_log.csv')
training_sets.append('./Edge Recovery Data/driving_log.csv')

# TODO: Train on Track 2
#training_sets.append('./Track 2 Data/driving_log.csv')

# For the left and right images, use a correction factor to teach the Model to steer away from the boundary
steering_correction = 0.2

# Control Parameter to decide on whether to use data with steering angle = 0 for training
exclude_zero_steering_angles = True

# Control Parameter to decide whether to augment the training data with horizontally flipped images
# Not needed as the training data provided by Udacity has good balance of + & ive steering angles
augment_training_data_with_flipped_data = False

# Size of each batch, passed to the model from the Generators
batch_size = 128

# Number of Epochs to run the model is set to 3, because after 3 Epochs did not see much gains
epochs = 3


# In[3]:


############################### Data Load Section ###############################

# CSV entries List to store the data from the CSV file
csv_list = []

# Load the entries from CSv file into the list
for training_data in training_sets:
    with open(training_data) as csv_data:
       csv_contents = csv.reader(csv_data)
       for entry in csv_contents:
          csv_list.append(entry)

# Remove first line header of csv i.e. Header
csv_list.pop(0)

# List of Camera Images and Steering Angles from the CSV file
camera_imgs = []
steering_angles = []

# For every entry in the CSV list, collect the images and steering angles
# If using the left and right images, use the steering_correction
for entry in csv_list:
    steering_angle_center = float(entry[3])
    if(exclude_zero_steering_angles == True and steering_angle_center == 0.0):
        continue
    else:
        # Read the center, left and right images
        center_img = cv2.imread(entry[0],cv2.COLOR_BGR2RGB)
        left_img   = cv2.imread(entry[1],cv2.COLOR_BGR2RGB)
        right_img  = cv2.imread(entry[2],cv2.COLOR_BGR2RGB)
        
        # Update the steering angles for left and right images, according to correction
        steering_angle_left   = steering_angle_center + steering_correction
        steering_angle_right  = steering_angle_center - steering_correction
       
        # Collate the images and the steering angles in correct order
        camera_imgs.extend([center_img,left_img,right_img])
        steering_angles.extend([steering_angle_center,steering_angle_left,steering_angle_right])

# Convert data into numpy arrays
camera_imgs = np.array(camera_imgs)
steering_angles = np.array(steering_angles)

# Check if we need to augment data with Flipped Images
if(augment_training_data_with_flipped_data == True):
    camera_imgs = np.concatenate((camera_imgs, np.flipud(camera_imgs)))
    steering_angles = np.concatenate((steering_angles, np.negative(steering_angles)))

X_samples = camera_imgs
y_samples = steering_angles

print("Training Data - Input Data  Size:",len(X_samples),"Shape:",X_samples[0].shape)
print("Training Data - Output Data Size:",len(y_samples),"Shape:",y_samples[0].shape)


# In[4]:


############################### Data Vizualization Section ###############################

# To display a grid of images
def show_images(X, end, total, images_per_row = 30, images_per_col = 15,
                H = 20, W = 1, its_gray = False):    
    number_of_images = images_per_row * images_per_col
    figure, axis = plt.subplots(images_per_col, images_per_row, figsize=(H, W))
    figure.subplots_adjust(hspace = .2, wspace=.001)
    axis = axis.ravel()
    
    for i in range(number_of_images):
        index = random.randint(end - total, end - 1)
        image = X[index]
        axis[i].axis('off')
        if its_gray:
          axis[i].imshow(image.squeeze(axis = 2), cmap='gray')
        else:
          axis[i].imshow(image.squeeze())


# In[5]:


# To plot bar graphs of label distribution
def plot_graph(data, name):
    # Convert numpy array to list
    label_list = data
    # List the labels
    labels = range(len(data))
    # Count the number of samples for each label
    #label_count = [label_list.count(i) for i in labels]
    
    fig = plt.figure(figsize=(30,20))
    
    # Plot the bar graph
    plt.bar(labels, label_list, width=16, figure=fig)
    plt.xlabel(name)

    axes = plt.gca()
    axes.set_ylim([-1.5,1.5])
    axes.set_xlim([0,len(labels)])

    plt.show()


# In[6]:


# Split the training data into training set and validation set 
X_train, X_valid, y_train, y_valid = train_test_split(X_samples, y_samples, test_size = 0.2)
print('Training Set  :',X_train.shape)
print('Validation Set:',X_valid.shape)


# In[7]:


############################### Data Preprocessing & Augmentation Section ###############################

# Image Preprocessing : To
def preprocess_image_data(input):
    # Convert to Grayscale
    img = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)        
    # Blur the image
    bimg = cv2.GaussianBlur(img,(9,9),0)
    # Localized Threshholding of Pixels with respect to its regions to highlight features liked road boundaries
    # Objective is to try to capture only data of relevance in the input image
    img = cv2.adaptiveThreshold(bimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,9,2) # 11 for 3
    # Crop the image to remove all the unwanted data in the image like the sky, surroundings and car bonnet
    x = img[70:130,:]
    # Normalize the data by Minimizing the range of values of each pixel by scaling
    x = (x / 255.).astype(np.float32)
    # Normalize contrast by equalizing histogram of images
    x = exposure.equalize_adapthist(x)
    # Option of Highlighting edges and blurring non edges better with bilaterFilter
    #x[i] = cv2.bilateralFilter(x[i],9,75,75)
    # Add the "THIRD" dimension
    x = np.expand_dims(x, axis=3)
    return x

# todo : Preprocessing the steering angles to improve response
#def preprocess_steering_data(y):
    # Smoothening OR Augment the angles

# Vizualize Raw Data v.s Processed Data
X_viz = []
p_X_viz = []

# Number of images to visualize
num_of_viz_imgs = 20

# Select Random set of input images
r_indices = random.sample(range(len(X_train)),num_of_viz_imgs)
for i in r_indices:
    X_viz.append(X_train[i])
    p_X_viz.append(preprocess_image_data(X_train[i]))

# Visualize the raw image dataset
show_images(X_viz, len(X_viz), len(X_viz), images_per_row = 4, images_per_col = 5, H = 40, W = 20)
    
# Visualize the processed image dataset
show_images(p_X_viz, len(p_X_viz), len(p_X_viz), images_per_row = 4, images_per_col = 5, H = 40, W = 20, its_gray=True)


# In[8]:


# Visuzalie the output labels to better understand distribution and update with new info

# todo : Plot each unique steering angle only once

# Visualize the Steering angle distribution across the dataset
#plot_graph(y_samples,'Training Data Steering Angle Distribution')

# Plot a histogram of the steering angles
plt.hist(y_samples, bins=20, range=(-1.0,1.0), align=('mid'))
plt.show()


# In[9]:


# Prepare Batch Generator for the feeding the training data to the model in batches

def generator(x,y,batch_size):
    num_samples = len(x)
   
    while 1: # Loop forever so the generator never terminates
        shuffle(x,y)
        
        for offset in range(0, num_samples, batch_size):            
            batch_x = np.zeros((batch_size,60,320,1))
            #batch_y = np.zeros(batch_size)

            batch_samples = x[offset:offset+batch_size]
            batch_labels = y[offset:offset+batch_size]
      
            for index in range(len(batch_samples)):
                batch_x[index] = preprocess_image_data(batch_samples[index])

            yield(batch_x[:len(batch_samples)], batch_labels)

training_generator = generator(X_train, y_train,batch_size)
validation_generator = generator(X_valid, y_valid,batch_size)


# In[10]:


############################### Deep NNet Model Section ###############################

# Scaled Down NVidia CNN for Behavioral CLoning
model = Sequential()
model.add(Convolution2D(24,3,3,subsample=(2,2),activation="relu",input_shape=(60,320,1)))
model.add(Convolution2D(36,3,3,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,3,3,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#ToDo: Use Callbacks to save the best performing Epoch


# In[11]:


############################### Model Training Section ###############################

model.compile(loss='mse', optimizer='adam')
# fits the model on batches with real-time data augmentation:
model.fit_generator(training_generator,samples_per_epoch=len(X_train),                    validation_data=validation_generator,nb_val_samples=len(X_valid),                    nb_epoch=epochs)
model.save('model.h5')
model.summary()


# In[12]:


# Vizualize the Outputs of Model Layers
# ToDo : Improve the drawing of weights and add more layers
from keras import backend as K

def get_layer_output(layer_nb):
    return K.function([model.layers[0].input],[model.layers[layer_nb].output])

def draw_sample_weight_channels(layer_output, layer_depth, x, y):
    # Collect Images
    layer_imgs = []
    for i in range(4):
        index = int(layer_depth/(4+i+1))
        layer_imgs.append(layer_output[:,:,:,index].reshape([x,y]))
    
    # Format arrays
    layer_imgs = np.array(layer_imgs)
    layer_imgs = np.expand_dims(layer_imgs, axis=3)

    # Visualize the processed image dataset
    show_images(layer_imgs, 4, 4, images_per_row = 2, images_per_col = 2, H = 40, W = 20, its_gray=True)
    
# Choose a random image for Layer Weight Visualization
img_index = random.randint(0,len(X_train))
eval_img = X_train[img_index]
# Pre-Process the image
x = preprocess_image_data(eval_img)

# Draw the image next to each other
plt.imshow(eval_img)
plt.title('Original Input Image - Steering Angle : '+str(y_train[img_index]))
plt.show()

plt.imshow(np.squeeze(x,axis=2), cmap='gray')
plt.title('Pre-Processed Input Image - Steering Angle : '+str(y_train[img_index]))
plt.show()

# Get Weights of the Convolutional Layer - 1
x = x.reshape(1, 60, 320, 1)
layer_1_output = get_layer_output(layer_nb = 1)([x])[0]
print('Layer 1 Weight:',layer_1_output.shape)

# Draw the visualizations, 4 samples from each layer weight
draw_sample_weight_channels(layer_1_output,layer_1_output.shape[3],layer_1_output.shape[1],layer_1_output.shape[2])

