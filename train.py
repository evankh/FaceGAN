import torch #To install torch use "pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
import inspect
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from keras import backend
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, LSTM, RepeatVector, Lambda
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import sys, glob
import os
import pytest
import argparse
import model

sgd = optimizers.SGD(lr=0.0002, momentum=0.0, nesterov=False)
#for more optimizers look at https://keras.io/optimizers/
# implementation of wasserstein loss function
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)
#BCE loss function, alternative to wasserstein
loss = nn.BCELoss()
# Total number of epochs to train
num_epochs = 100
num_test_samples = 16
test_noise = model.noise(num_test_samples)

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    sgd.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = model.discriminator(real_data)
    # Calculate error and backpropagate
    error_real = wassrstein_loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = model.discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    sgd.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    sgd.zero_grad()
    # Sample noise and generate fake data
    prediction = model.discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    sgd.step()
    # Return error
    return error

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data): #Iterate through data
        fake_data = model.generator(noise(N)).detach()
        N = real_batch.size(0)
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
        fake_data = model.generator(noise(N))
        g_error = train_generator(g_optimizer, fake_data)
