from PIL import Image
import numpy as np
import glob
import os
import cv2
import time
from random import randint
import matplotlib.pyplot as plt
import pyautogui


def standardized_images(images): #converts images list to array
    size = np.zeros([500,500,3])
    standardized_list = []
    first_time = True
    for i in range(len(images)):
        x, y, z = images[i].shape
        x_diff = 500 - x
        y_diff = 500 - y
        assert x_diff >= 0, 'too fat'
        assert y_diff >= 0, 'too tall'
        if y_diff < 0 or x_diff < 0:
            print('here')
        standardized_image = cv2.copyMakeBorder(images[i], 0, x_diff,0, y_diff, cv2.BORDER_CONSTANT)
        vector = np.ndarray.flatten(standardized_image)
        if first_time is True:
            global array
            array = vector
            first_time = False
        else:
            array = np.dstack((array, vector))
        if i%100 == 0:
            print(i)
            pyautogui.move(0, 1)
    product = np.reshape(array, (750000, -1))
    
    return product


def truth_label(num_images, cat):
    Y = np.arange(num_images)
    if cat is True:
        Y.fill(1)
    elif cat is False:
        Y.fill(0)
    else:
        print("Truth label error")
    Y = np.reshape(Y, (1, num_images))
    
    return Y


def process(array_1, array_2, Y_1, Y_2, set_size):
    train_size = 2 * round(set_size/3)
    test_size = set_size - train_size
    X = np.concatenate((array_1, array_2), axis = 1)
    Y = np.concatenate((Y_1, Y_2), axis = 1)

    rng = np.random.default_rng(seed = 3)
    rng.shuffle(X, axis = 1)

    rng = np.random.default_rng(seed = 3)
    rng.shuffle(Y, axis = 1)
    X_train = X[:, :train_size]
    Y_train = Y[:, :train_size]
    X_test = X[:, :test_size]
    Y_test = Y[:, :test_size]
    
    return X_train, Y_train, X_test, Y_test

start = time.time()

cats = []
dogs = []
set_size = 1000
count = 0
num_images = set_size

for file in glob.glob('kagglecatsanddogs_3367a/PetImages/Cat/*.jpg'):
    im = Image.open(file)
    im = np.array(im)
    test = np.reshape(im, (im.shape[0], im.shape[1], -1))
    if test.shape[2] == 3:
        cats.append(im)
    set_size -= 1
    if set_size <= 0:
        print("Cats done")
        break

for i in range(len(cats)):
    im = cats[i]
    test = np.reshape(im, (im.shape[0], im.shape[1], -1))
    assert test.shape[2] == 3, "failed"


set_size = num_images
for file in glob.glob('kagglecatsanddogs_3367a/PetImages/Dog/*.jpg'):
    im = Image.open(file)
    im = np.array(im)
    im = np.array(im)
    test = np.reshape(im, (im.shape[0], im.shape[1], -1))
    if test.shape[2] == 3:
        cats.append(im)
    set_size -= 1
    if set_size <= 0:
        print("Dogs done")
        break
    
standardized_cats = standardized_images(cats)
standardized_dogs = standardized_images(dogs)
Y_cats = truth_label(num_images, True)
Y_dogs = truth_label(num_images, False)

X_train, Y_train, X_test, Y_test = process(standardized_cats,standardized_dogs, Y_cats, Y_dogs, num_images)
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)
np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)
end = time.time()
time = end - start
print('Runtime: ',time)
