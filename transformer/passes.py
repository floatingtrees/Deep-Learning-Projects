# This program uses layers to build a simple feedforward network for cat and dog image recognition

import numpy as np
import pickle
import time
import layers

X_train = np.load('X_train_confirmed.npy').T.astype(np.float32) / 255
Y_train = np.load('Y_train_confirmed.npy').T.astype(np.float32)
X_test = np.load('X_test_confirmed.npy') / 255
print(X_test.shape)
Y_test = np.load('Y_test_confirmed.npy')      
start = time.time()

y = Y_train


def sigmoid(x):
    return 1/(np.exp(-x) + 1)

def sigmoid_backward(x):
    v = sigmoid(x)
    return np.multiply(v, (1 - v))


norm = layers.BatchNorm()
norm.build(X_train)
a = norm.call(X_train)

print(np.mean(a), np.std(a))


dense1 = layers.Dense(64)
dense1.build(a)
a = dense1.call(a)

drop = layers.Dropout(0.5)
drop.build(a)
a = drop.call(a)

dense2 = layers.Dense(1, activation = "sigmoid")
dense2.build(a)
a = dense2.call(a)

prev_loss = 100
epochs = 50
for i in range(epochs):
    a1 = norm.call(X_train)
    a2 = dense1.call(a1)
    a3 = drop.call(a2)
    a4 = sigmoid(dense2.call(a3))
    #Cost
    c = np.log(a4)
    c = np.multiply(y, c)
    loss = - np.sum(c) / y.shape[0]
    print("LOSS: ", loss)
    print("EPOCH: ",prev_loss - loss)
    prev_loss = loss
    dcda4 = loss * 0.01 * a4
    dcdz4 = dcda4 * sigmoid_backward(dcda4)
    dcda3 = dense2.compute_gradients(dcdz4)
    dcda2 = drop.compute_gradients(dcda3)
    dcda1 = dense1.compute_gradients(dcda2)
    dcdx = norm.compute_gradients(dcda1)
    dense2.update()
    drop.update()
    dense1.update()
    norm.update()








print("\n\n\n\n\nTime taken: ", time.time() - start)
