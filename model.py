import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
import csv
import math
from sklearn.model_selection import train_test_split

steering_correction = .15
steering_threshold = .25
batch_size = 40
num_epoch = 2

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

del(lines[0]) 
train_set, valid_set = train_test_split(lines, test_size=0.2, random_state=100)

    
train_set, valid_set = train_test_split(lines, test_size = 0.2)

def shadow_augmentation(image):
	'''refer to: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
	'''
	height, width, _ = image.shape
	imgtop_y = width*np.random.uniform() # 320
	imgtop_x = 0
	imgbot_y = width*np.random.uniform()
	imgbot_x = height # 160
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	shadow_mask = 0*image[:,:,1]
	mask_X = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
	mask_Y = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
	shadow_mask[((mask_X - imgtop_x)*(imgbot_y - imgtop_y) - (imgbot_x - imgtop_x)*(mask_Y - imgtop_y) >=0)] = 1
	if np.random.randint(2) == 1:
		condition_1 = shadow_mask == 1
		condition_0 = shadow_mask == 0
		if np.random.randint(2) == 1:
			image[:,:,1][condition_1] = image[:,:,1][condition_1]*.5 # random brightness
		else:
			image[:,:,1][condition_0] = image[:,:,1][condition_0]*.5 
	image = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
	return image 


def brightness_process(image):
	image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # convert from RGB to HSV color
	brightness_random =  .23 + np.random.uniform() # random brightness 
	image[:,:,2] = image[:,:,2]*brightness_random  # add brightness to the image
	image[:,:,2][image[:,:,2]>255] = 255		   # keep image at 255 if greater than that 
	image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB) # convert back to RGB from HSV
	return image

def process_images(center_path, left_path, right_path, steering, images, steering_set):
	# center
	center_image = cv2.imread('data/IMG/' + center_path.split('/')[-1]) # extract the image from the path
	center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)		# convert BGR to RGB
	center_image = brightness_process(center_image)						# brightness process
	center_image = shadow_augmentation(center_image)					# shadow augmentation 
	images.append(center_image)											
	steering_set.append(steering)
	# left
	left_image = cv2.imread('data/IMG/' + left_path.split('/')[-1])
	left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
	left_image = brightness_process(left_image)
	left_image = shadow_augmentation(left_image)
	images.append(left_image)
	steering_set.append(steering + steering_correction)
	# right
	right_image = cv2.imread('data/IMG/' + right_path.split('/')[-1])
	right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
	right_image = brightness_process(right_image)
	right_image = shadow_augmentation(right_image)
	images.append(right_image)
	steering_set.append(steering - steering_correction)
	if abs(steering) > steering_threshold:			# if steering is greater than steering_threshold, then flip the image.
		images.append(cv2.flip(center_image, 1)) 	# flip images (augmentation)
		steering_set.append(-steering)
		images.append(cv2.flip(left_image, 1))		
		steering_set.append(-(steering + steering_correction))
		images.append(cv2.flip(right_image, 1))		
		steering_set.append(-(steering - steering_correction))

def generators(datasets, batch_size=batch_size):
	while True: 
		shuffle(datasets) # shuffle the datasets
		for offset in range(0, len(datasets), batch_size):
			batch_lines = datasets[offset : offset + batch_size]
			images = []
			steering_set = []
			for line in batch_lines:
				steering = float(line[3]) 
				process_images(line[0], line[1], line[2], steering, images, steering_set)
			X_train = np.asarray(images)
			y_train = np.asarray(steering_set)
			yield shuffle(X_train, y_train, random_state=100)
            

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# keras libs
from keras import backend
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
# misc. import
import numpy as np 
import csv
import cv2

def nvidia_model():
	model = Sequential()
	model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25), (0,0))))  # 70-top, 25-bottom
	model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu'))
	model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu'))
	model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
	model.add(Convolution2D(64,3,3, activation='elu'))
	model.add(Convolution2D(64,3,3, activation='elu'))
	model.add(Flatten())
	model.add(Dense(100, activation='elu'))
	#model.add(Dropout(0.5))
	model.add(Dense(50, activation='elu'))
	#model.add(Dropout(0.5))
	#model.add(Dense(20))
	#model.add(Dropout(0.5))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))
	model.summary()
	return model

def run_the_network():
    # architectural network (nvidia-based)
    model = nvidia_model()
    # get data
    #training_set, validation_set = get_log()
    # train the model and fit generator
    train_set_generator = generators(train_set, batch_size=batch_size)
    valid_set_generator = generators(valid_set, batch_size=batch_size)
    # params for fit generator
    samples_per_epoch = (3*len(train_set)//batch_size)*batch_size
    nb_val_samples = len(valid_set)
    
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_set_generator, samples_per_epoch= samples_per_epoch, validation_data=valid_set_generator, nb_val_samples=nb_val_samples, nb_epoch=num_epoch, verbose=1)

    model.save('model_ali.h5')
    
run_the_network()

backend.clear_session() 
