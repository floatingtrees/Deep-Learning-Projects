import numpy as np
import pickle
import time

# matrix of word embeddings, use indexing to find the proper embedding


class Embedding:
    def __init__(self, vocab_size, features):
        self.vocab_size = vocab_size
        self.features = features
        
    def build(self):
        self.weights = np.random.rand(vocab_size, features).astype(np.float32)

    def call(self, inputs): # inputs is in shape (batch size, tokens)
        embedded_vectors = np.zeros((inputs.shape[0], inputs.shape[1], self.features))
        # pretty sure this works
        index = np.array(inputs)
        sequence = np.array(self.weights[index, :])
        return sequence


    def backprop(self, grads):
        pass

class Dense:
    def __init__(self, neurons, activation):
        self.neurons = neurons
        self.activation = activation
        self.biases = np.zeros((1, neurons), dtype = np.float32)
    def build(self, dummy_input):
        self.weights = np.random.rand(dummy_input.shape[-1], neurons).astype(np.float32)
    def call(self, inputs):
        return np.matmul(inputs, self.weights) + self.biases

class Model:
    def __init__(self):
        self.Dense1 = self.
