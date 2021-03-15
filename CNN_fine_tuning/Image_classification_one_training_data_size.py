import glob
import os
import math
from random import shuffle
import time
import numpy as np
import random
import cv2 as cv
import matplotlib.pyplot as plt
from keras.utils import np_utils

def get_image_files(root_dir, img_types):
    #os.walk creates 3-tuple with (dirpath, dirnames, filenames)

    # Get all the root directories, subdirectories, and files
    full_paths = [x for x in os.walk(root_dir)]
    imgs_temp = [os.path.join(ds,f) for ds,_,fs in full_paths for f in fs if f]

    # Filter out so only have directories with .jpg, .tiff, .tif, .png, .jpeg
    imgs = [j for j in imgs_temp if any (k in j for k in img_types)]
    return imgs

def get_dimensions(files):
    # Set starting points for min and max dimensions
    min_height, min_width = 10000, 10000
    max_height, max_width = 0, 0

    for f in files:
        # Read in images
        img = cv.imread(f) # Read in images
        h,w = img.shape[:2] # get height and width

        # Update min and max values, if necessary
        if h < min_height:
            min_height = h
        if h > max_height:
            max_height = h
        if w < min_width:
            min_width = w
        if w > max_width:
            max_width = w

    return min_height, min_width, max_height, max_width

def make_labels(files):
    # Assume input is a list of complete file paths.
    # Count the number of unique directory names that are immediate parent of the files.
    # Order the directory names alphabetically from a-z, and associate labels accordingly.
    set_temp = {x.split('/')[-2] for x in files} #doing as set to get only unique values
    list_temp = list(set_temp) #Change to list so can interate over it
    list_new = sorted(list_temp) #Alphabetizing
    label_dict = {list_new[x]:x for x in range(len(list_new))} #create dictionary with category:index

    return label_dict

def make_variable_train_fixed_val(files, labels, num_train, num_val):
    """
    param files: list of all images
    type files: str
    param labels: a mapping of image class to a class value
    type labels: dict
    param num_train: number of training examples
    type num_train: int
    param num_val: number of validation examples
    type num_val: int
    return: a list of training and validation images
    """
    train=[]
    valid = []
    for key in labels: #going through each key
        temp = [f for f in files if key in f] #getting all files in a specific category (ie key)
        temp = sorted(temp)
        if len(temp) < num_train + num_val:
            raise ValueError('num_train + num_val exceeds total number of images for {}'.format(key))
        train.extend(temp[:num_train]) #training data set
        valid.extend(temp[-num_val:]) # validation data set
    return train, valid

def make_train_val(files, labels):
    train=[]
    valid = []
    train_labels_name = []
    valid_labels_name = []
    train_prop = 0.8 #proportion of data set that will be training
    for key in labels: #going through each key
        temp = [f for f in files if key in f] #getting all files in a specific category (ie key)
        temp = sorted(temp)
        train.extend(temp[:math.ceil(train_prop*len(temp))]) #training data set
        valid.extend(temp[math.ceil(train_prop*len(temp)):]) # validation data set
    train_labels_name = [x.split('/')[-2] for x in train]
    valid_labels_name = [x.split('/')[-2] for x in valid]
    return train, valid, train_labels_name, valid_labels_name

def make_train_val_test(files, labels):
    train=[]
    valid = []
    test =[]
    train_labels_name = []
    valid_labels_name = []
    test_labels_name = []
    train_prop = 0.6 #proportion of data set that will be training
    val_prop = 0.2 #proprotion of dataset that is validation
    for key in labels: #going through each key
        temp = [f for f in files if key in f] #getting all files in a specific category (ie key)
        lower_prop = math.ceil(train_prop*len(temp))
        train.extend(temp[:lower_prop]) #training data set
        valid.extend(temp[lower_prop:lower_prop+math.ceil(val_prop*len(temp))]) # validation data set
        test.extend(temp[lower_prop+math.ceil(val_prop*len(temp)):])
    train_labels_name = [x.split('/')[-2] for x in train]
    valid_labels_name = [x.split('/')[-2] for x in valid]
    test_labels_name =  [x.split('/')[-2] for x in test]
    return train, valid, test, train_labels_name, valid_labels_name, test_labels_name

def get_batches(files, label_map, batch_size, resize_size, num_color_channels, augment=False, predict=False, do_shuffle=True):
    if do_shuffle:
        shuffle(files)
    count = 0
    num_files = len(files)
    num_classes = len(label_map)

    batch_out = np.zeros((batch_size, resize_size[0], resize_size[1], num_color_channels), dtype=np.uint8)
    labels_out = np.zeros((batch_size,num_classes)) #one-hot labeling, which is why have num_classes num of col.

    while True: # while True is to ensure when yielding that start here and not previous lines

        f = files[count]
        img = cv.imread(f)

        # Resize
        # First resize while keeping aspect ratio
        rows,cols = img.shape[:2] # Define in input num_color_channels in case want black and white
        rc_ratio = rows/cols
        if resize_size[0] > int(resize_size[1]*rc_ratio):# if resize rows > rows with given aspect ratio
            img = cv.resize(img, (resize_size[1], int(resize_size[1]*rc_ratio)))#NB: resize dim arg are col,row
        else:
            img = cv.resize(img, (int(resize_size[0]/rc_ratio), resize_size[0]))

        # Second, pad to final size
        rows,cols = img.shape[:2] #find new num rows and col of resized image
        res = np.zeros((resize_size[0], resize_size[1], num_color_channels), dtype=np.uint8)#array of zeros
        res[(resize_size[0]-rows)//2:(resize_size[0]-rows)//2+rows,
            (resize_size[1]-cols)//2:(resize_size[1]-cols)//2+cols,:] = img # fill in image in middle of zeros

        # Augmentation
        if augment:
            rows,cols = res.shape[:2]
            # calculates affine rotation with random angle rotation, keeping same center and scale
            M = cv.getRotationMatrix2D((cols/2,rows/2),np.random.uniform(0.0,360.0,1),1)
            # applies affine rotation
            res = cv.warpAffine(res,M,(cols,rows))

        # Change to gray scale if input argument num_color_channels = 1
        if num_color_channels == 1:
            res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)# convert from bgr to gray
            res = res[...,None] # add extra dimension with blank values to very end, needed for keras

        batch_out[count%batch_size,...] = res # put image in position in batch, never to exceed size of batch

        for k in label_map.keys():
            if k in f: #if a category name is found in the path to the file of the image
                labels_out[count%batch_size,:] = np_utils.to_categorical(label_map[k],num_classes) #one hot labeling
                break

        count += 1
        if count == num_files: # if gone through all files, restart the counter
            count = 0
        if count%batch_size == 0: #if gone through enough files to make a full batch
            if predict: # i.e., there is no label for this batch of images, so in prediction mode
                yield batch_out.astype(np.float)/255.
            else: # training
                yield batch_out.astype(np.float)/255., labels_out

# Get full paths to all classification data
# Data is assumed to reside under the directory "root_dir", and data for each class is assumed to reside in a separate subfolder
root_dir = '/home/dtaniguchi/Desktop/SPCP2_Images_450'
img_types=['.jpg', '.tiff', '.tif', '.png', '.jpeg']
save_dir = 'tmp' #TODO: change

files = get_image_files(root_dir, img_types)
print('number of files is ',len(files))
print('example file names are ', files[0:4])

# Get the dimension range of the data for informational purposes
minh,minw,maxh,maxw = get_dimensions(files)
print('Over all images - minimum height: {}, minimum width: {}, maximum height: {}, maximum width:{}'.format(minh,minw,maxh,maxw))

# Assign numerical labels to categories - the number of categories is equal to the number of subfolders
label_map = make_labels(files)
print(label_map)

from keras.applications.inception_v3 import InceptionV3 #--[don't need if running Xception]
#from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D,  Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD

def classifier(n_dense_units, dropout=0.0):
    #input_shape taken from get_dimensions in Jupyter notebook Image_classification
    # create the base pre-trained model

    #base_model = Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=(880,920,3), pooling=None)

    base_model = InceptionV3(weights='imagenet', include_top=False) #--[don't need if using Xception]

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = Dense(n_dense_units, activation='relu')(x)
    x = Dropout(dropout)(x)

    # and a logistic layer -- let's say we have x classes--determined by len(label_map)
    predictions = Dense(len(label_map), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    opt = SGD(lr=0.001, momentum=0.9, decay=1e-6) #Adam(lr=0.00005)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

# Code in this cell taken from
#https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
# train the model on the new data for a few epochs
# initialize the number of epochs and batch size
from time import time

BS = 8
EPOCHS = 1000

dataGen_train = ImageDataGenerator(featurewise_center=False, samplewise_center=False,
                    featurewise_std_normalization=False, samplewise_std_normalization=False,
                    rotation_range=360, width_shift_range=0.2, height_shift_range=0.2, 
                    zoom_range=0.5, fill_mode='constant', cval=0, horizontal_flip=True,
                    vertical_flip=True, rescale=None)


"""
Following code snippets are for splitting a dataset into a disjoint set for training and validation.
Training data size is fixed and does not vary
"""
range_dense_units = [8, 16] #[1,2,4,8]
range_dropout = [0.4] #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
num_trials_per_permutation = 5

# split dataset into train/val sets at 80%/20%
train_files, val_files, train_labels_name, val_labels_name = make_train_val(files, label_map)

# Get batch
batch_gen = get_batches(train_files,label_map,batch_size=len(train_files),resize_size=[150, 150],num_color_channels=3)
val_gen =   get_batches(val_files,label_map,batch_size=len(val_files),resize_size=[150, 150],num_color_channels=3)
train_data, train_labels = next(batch_gen)
val_data, val_labels_oh = next(val_gen)

for num_dense_units in range_dense_units: # cycle through a range of intermediate dense units
    for dropout in range_dropout: # cycle through a range of dropout values
        for nt in range(num_trials_per_permutation):
            print('Trial: {}, num_train_examples_per_class: {}, num_dense_units: {}, dropout: {}'.format(
                nt, len(train_files), num_dense_units, dropout))

            ES = EarlyStopping(monitor='val_loss', patience=20, verbose=0)
            MC = ModelCheckpoint(
                save_dir + '/model_dense-units={}_dropout={}_trial={}.h5'.format(num_dense_units, dropout, nt),
                monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1
            )
            model = classifier(num_dense_units)

            sta_time = time()
            model.fit_generator(dataGen_train.flow(train_data, train_labels, batch_size=BS),
                epochs=EPOCHS,
                steps_per_epoch=len(train_files) // BS,
                validation_data=(val_data, val_labels_oh),
                callbacks=[ES, MC],
                verbose=0)

            del model
            model = load_model(save_dir + '/model_dense-units={}_dropout={}_trial={}.h5'.format(num_dense_units, dropout, nt))
            val_acc = model.evaluate(val_data, val_labels_oh, batch_size=BS, verbose=0)
            results = 'Trial: {}, num_training_per_class: {}, num_dense_units: {}, ' \
                'dropout: {}, validation (loss, acc): ({:.2g}, {:.2g}), time to train: {}s\n'.format(
                nt, len(train_files), num_dense_units, dropout, *val_acc, int(time()-sta_time))

            print(results)
            with open('Image_classification_one_training_data_size_results_{}_train_examples.txt'.format(len(train_files)), 'a') as f:
                f.write(results) # write results to file

            model.save(save_dir + '/model_dense-units={}_dropout={}_val-acc={:.2g}.h5'.format(num_dense_units, dropout, val_acc[1]))

print('Done')
