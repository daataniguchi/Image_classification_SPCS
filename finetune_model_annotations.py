# -*- coding: utf-8 -*-
"""
finetune_model

Finetune a convnet from a pretrained model using training and validation data

This is based heavily on The Transfer Learning Tutorial from pytorch:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

"""
# License: BSD
# Author(s): Paul Roberts, Eric Orenstein
###import things
from __future__ import print_function, division ### future is used to be accustomed to incompatible changes or new key words
import argparse ### easier to write user firendly command line interfaces, generates help and usage messages
import torch ###framework with wise support for machine learning algorithms
import torch.nn as nn ###no nn = neural network from scratch therefore import an neural network
### "as" indicates it will be referred as that throughout the code
import torch.optim as optim ###a package that has optimization algotrithms
from torch.optim import lr_scheduler ###learning rate scheduling, used to adjust the hyperparameter of the learning rate model
import numpy as np ###used for working with arrays
import torchvision ###consists of  popular datasets, model architectures and common image transformation for computer vision. Spedifies the package used to load images
from torchvision import datasets, models, transforms ### pulling from torchvision
import matplotlib.pyplot as plt ###popular visualization package it manipulates elements of a figure
import time ###handles time related tasks
import os ###provides functions for interacting with the operating system
import copy ###model for shallow or deep copy operations

"""
train_model
finetunes the net based on input params and train/val data
#####we switched frameworks to pytorch 
#### tain=training data
#### val=validation data 
"""
###link for definitions for torch.nn https://pytorch.org/docs/stable/nn.html#crossentropyloss
###go resource is on youtube. Pytorch tutorials by python engineer (I used 15 the most for transfer learning)

###function to train the model
def train_model(model, criterion, optimizer, scheduler,
    dataloaders, classes, num_epochs=25): ###number of epochs that will be run (in this case 24), criterion=compute gradiants according to given loss function,

    since = time.time() ###since is the time as floating number expressed in sec since the epoch

    best_model_wts = copy.deepcopy(model.state_dict()) ###weight loading by deep copying a model (deep copy function that means any changes made to a copy do not change the orginal)
    best_acc = 0.0 ###best_acc start at zero

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1)) ###print stament on which epoch has run (i.e. Epoch 20/24)
        print('-' * 10) ###print - 10 times (just to seperate the epochs (organization))
###epoch =1 forward and backward pass of all training samples

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train': ###if epoch in training phase
                scheduler.step() ###needs to be called every epoch bc if not learning rate wont change and will stay at the inital value
                model.train()  # Set model to training mode
            else: ###if epoch not in train phase therefore in val phase
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0 ###starting running loss
            running_corrects = 0 ###starting running corrects

            class_corrects = np.zeros(len(classes)) ###numbpy zero function making numbpy array with only zeros(empty array) to the length of classes to hold the class corrects
            class_counts = np.zeros(len(classes)) ###same as above (to the length of the classes to hold the class count

            # Iterate over data.  ###iteration=number of passes, samples/batchsize
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) ###returns a copy of inputs that resides on device and to use "on device" need to assign variable
                labels = labels.to(device) ###same as above but for labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'): ###clear intermediate values for evaluation important for backprop
                    outputs = model(inputs) ###define outputs
                    _, preds = torch.max(outputs, 1) ###torch.max returns max value in the tensor
                    loss = criterion(outputs, labels) ###calculate loss pass the output with labels

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() ###computes dloss/dx for every parameter
                        optimizer.step() ###updates the value of x (parameters)

                # statistics (calc loss and corrections
                running_loss += loss.item() * inputs.size(0) ###calculating the running loss
                running_corrects += torch.sum(preds == labels.data) ###calculating the running corrections

                for i,p in enumerate(preds): ###???
                    class_corrects[labels.data[i]] += p == labels.data[i]
                    class_counts[labels.data[i]] += 1

####EPOCH loss and acc calculation
            epoch_loss = running_loss / dataset_sizes[phase] ###calcuclting the epoch loss
            epoch_acc = running_corrects.double() / dataset_sizes[phase] ###calculate the epoch acc

            class_acc = class_corrects/class_counts ###calculate the class acc


            print('{} Loss: {:.4f} Acc: {:.4f}'.format( #### the train loss and acc will be printed
                phase, epoch_loss, epoch_acc))

            for i,c in enumerate(classes): ###enumerate=values printed along their index
                print('{} Class: {} Acc {:.4f}'.format(phase,c,class_acc[i]))


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) ###weight loading by deep copying a model

        print()

    time_elapsed = time.time() - since ###how long the training took
    print('Training complete in {:.0f}m {:.0f}s'.format( ###print and calculate how long the training took
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc)) ###print the best val acc

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
###Function to training model is OVER

###uses argparse to help write user friendly comandline interface, auto genterates useful messages https://docs.python.org/3/library/argparse.html
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Finetune a convnet') ###create parser, hold info necessary to parse command line

    parser.add_argument('data_dir',metavar='data_dir',help='path to train and val data') ###filling the argument parser with info about program

    parser.add_argument('--model',default='resnet34',choices=['resnet18','resnet34','squeezenet'],
        help='The type of model to finetune')

    parser.add_argument('--epochs',default=24,type=int,help='The number of training epochs')

    args = parser.parse_args()

    nepochs = args.epochs

    # Data augmentation and normalization for training
    # Just normalization for validation
    ###https://pytorch.org/docs/stable/torchvision/transforms.html
    data_transforms = {  ###applied to each batch set, randomizing, resizing/cropping, randomly flipping
        'train': transforms.Compose([ ###transforms for train, slight changes can be made if necessary (i.e random resizing)
            #transforms.RandomResizedCrop(224),
            transforms.Resize((224,224)), ###resize to a 224,224 patch
            transforms.RandomHorizontalFlip(), ###random horizontal flip
            transforms.RandomVerticalFlip(), ###random vertical flip
            transforms.RandomAffine(degrees=0,scale=(0.5, 2), shear=(-5, 5)), ###image keeping center invariant (degree=0, deactivate rotations, scale, scaling factor interval, shear(-,+) shear paralllel to x axis
            transforms.ToTensor(), ###convert to tensor (data holding unit of pytorch(kinda like a matrix))
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ###Normalize tensor with mean and SD WE provide (may need to change, not sure)
        ]),
        'val': transforms.Compose([ ###transform for val
            transforms.Resize((224,224)), ###resize to a 224,224 patch
            #transforms.CenterCrop(128), ###can add function to centercrop the image to 128
            transforms.ToTensor(), ###convert to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ###Normalize tensor with mean and SD (may need to change)
        ]),
    }
###loading data
    data_dir = args.data_dir ####pulling from a directory (put our files)

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), ###retrieve dataset from folder that has inside train and val folders and inside those 'x' and'y', organization process
                                              data_transforms[x]) ###transform dataset
                      for x in ['train', 'val']} ###for each value in train and val creating dirt which x is key
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, ####organization process determine batchsize (batchsize= number of training samples in one forward and backward pass), shuffle and num workers (each worker loads a single batch (has to do with memory))
                                                 shuffle=True, num_workers=4)   ####https://pytorch.org/docs/stable/data.html
                  for x in ['train', 'val']} ####loop through for each value
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']} ####length of the dataset for each batch
    class_names = image_datasets['train'].classes ###get class names by calling image_dataset.classes

    ###need to add a function to seperate folders (may already have a seperate code to do so)

    print(class_names) ###print the class names

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ###creation of device (cuda=plateform to use GPU therefore if cuda not aviable CPU used

    model_type = 'resnet18' ###store model type as resnet18

###this is the TRANSFER LEARNING(FineTuning technique= train model again only alitte,finetune all the weights based on new data and new last layer
###depending on which model are we using to transfer learning(in this code model_type= resnet18 stated above) adjuests the last fully connected layer https://medium.com/@14prakash/almost-any-image-classification-problem-using-pytorch-i-am-in-love-with-pytorch-26c7aa979ec4
    ###link to youtube video on transfer learning https://youtu.be/K0lWSB2QoIQ
    if model_type == 'resnet34': ### "Residual Networks" nn as a backbone for many computer vision tasks (with 34 layers)
        model_conv = models.resnet34(pretrained=True) ###pretained has already optimized weights on imagenet(for resnet) data
        ###exchange the last fully connected layer
        num_ftrs = model_conv.fc.in_features ###number of input features of last layer
        model_conv.fc = nn.Linear(num_ftrs, len(class_names)) ###the new layer and its assigned to the last layer
    if model_type == 'resnet18': ### resnet with 18 layers
        model_conv = models.resnet18(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    if model_type == 'squeezenet':###squeezenet is a small nn with fewer parameters, more easily fit into computer memeory and be transmitted over computer network https://towardsdatascience.com/review-squeezenet-image-classification-e7414825581a
        model_conv = models.squeezenet1_0(pretrained=True)
        # change the last Conv2D layer in case of squeezenet. there is no fc layer in the end.
        num_ftrs = 512
        model_conv.classifier._modules["1"] = nn.Conv2d(512, len(classes), kernel_size=(1, 1))
        # because in forward pass, there is a view function call which depends on the final output class size.
        model_conv.num_classes = len(classes)

    model_conv = model_conv.to(device) ###send the model_conv to the device if there is GPU support
    ###model reaches convergence when additional training will not improve the model (I am assuming model_conv=model convergence)

    criterion = nn.CrossEntropyLoss() ###criterion defined by croosentropyloss which combines logsoftmax and NLLLossin one single class

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_conv.parameters(), lr=0.002, momentum=0.9) ###define optimizer

    # Decay LR by a factor of 0.1 every 12 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=12, gamma=0.1) ###define schedulerevery 12 epoch the LR is multipled by o.1

    # DO the training and validation
    model_conv = train_model(model_conv, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, class_names, num_epochs=nepochs)


    # save the trained model
    torch.save(model_conv.state_dict(), ###model.state.dic saves the parameters
        os.path.join(data_dir,model_type+'_'+str(int(time.time()))+'_model_conv.pt'))
