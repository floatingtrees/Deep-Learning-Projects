import numpy as np
import pickle
import time

# matrix of word embeddings, use indexing to find the proper embedding

def softmax(x):
    x = np.exp(x)
    return x/np.sum(x)

def relu(x):
    return np.max(x, 0)

class Embedding:
    def __init__(self, vocab_size, features, positional_features):
        self.vocab_size = vocab_size
        self.features = features
        
    def build(self, dummy_input):
        self.weights = np.random.rand(vocab_size, features).astype(np.float32)
        self.positional_embedding = np.random.rand(dummy_input.shape[-2], positional_features).astype(np.float32)

    def call(self, inputs): # inputs is in shape (batch size, tokens)
        embedded_vectors = np.zeros((inputs.shape[0], inputs.shape[1], self.features))
        # pretty sure this works
        index = np.array(inputs)
        sequence = np.array(self.weights[index, :])
        a = np.concatenate(sequence, self.positional_embedding, axis = -1)
        self.a = a
        return a

    def backprop(self, grads):
        pass


class Dense:
    def __init__(self, neurons, activation = "relu"):
        self.neurons = neurons
        self.activation = activation
        
    def build(self, dummy_input):
        self.weights = np.random.rand(dummy_input.shape[-1], neurons).astype(np.float32)
        self.biases = np.zeros((1, neurons), dtype = np.float32)

    def call(self, inputs):
        z = np.matmul(inputs, self.weights) + self.biases
        if self.activation == "softmax":
            a = softmax(z)
        elif self.activation == "relu":
            a = relu(z)
        else:
            raise ValueError("Activation not found")
        self.a = a
        return a


class BatchNorm:
    def __init__(self, axis = -1, epsilon = 0.001):
        self.axis = axis
        self.epsilon = epsilon

    def build(self, dummy_input):
        self.samples = dummy_input.shape[0]
        self.moving_average = np.mean(dummy_input, axis = self.axis) + self.eposilon
        self.moving_std = np.std(dummy_input, axis = self.axis) + self.epsilon

    def call(self, inputs):
        a = inputs - self.moving_average / self.moving_std
        self.a = a
        return a

    def backprop(self):
        current_samples = a.shape[0]
        current_average = np.mean(a, axis = self.axis)
        current_std = np.std(a, axis = self.axis)
        self.moving_average = (self.moving_average * self.samples + current_average * current_samples) / (self.samples + current_samples) # might overflow
        self.moving_std = (self.moving_std * self.samples + current_std * current_samples) / (self.samples + current_samples) # might overflow

        return None # Fix for backprop

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        assert dropout_rate < 1, "unreasonable dropout rate: " + str(dropout_rate) # -1, 2

    def build(self):
        pass

    def call(self, inputs, training = True):
        if training:
            



class AttentionHead:
    def __init__(self):
        self.query_layer = Dense()
        self.key_layer = Dense()
        self.value_layer = Dense()

    def build(self, dummy_q, dummy_k, dummy_v = None):
        if dummy_v == None:
            dummy_v = dummy_k
        self.query_layer.build(dummy_q)
        self.key_layer.build(dummy_k)
        self.value_layer.build(dummy_v)

    def call(self, query, key, value = None): # figure out what to store
        q = np.matmul(query, self.query_layer)
        k = np.matmul(key, self.key_layer)
        if value == None:
            v = k
        else:
            v = np.matmul(value, self.value_layer)

        qk = np.matmul(q, np.swapaxes(k, -1, -2)) # transpose the matrix dimensions
        q_shape = q.shape
        k_shape = k.shape
        norm_factor = np.sqrt(q_shape[-1] * q_shape[-2] * k_shape[-1] * k_shape[-2])
        qk = qk/norm_factor
        qk = softmax(qk)
        a = np.matmul(qk, v)

        return a


class MultiHeadAttention:
    def __init__(self, num_heads):
        self.heads = {}
        self.Dense = Dense()
        for i in range(num_heads):
            self.heads[i] = AttentionHead()

    def build(self, dummy_q, dummy_k, dummy_v = None):
        for i in range(num_heads):
            self.heads[i].build(dummy_q, dummy_k, dummy_v) 
            self.Dense = 














class Model:
    def __init__(self):
        self.Dense1 = Dense(32, "relu")
