import numpy as np
import pickle
import time

# Make sure that necessary arrays are duplicated with ndarray.copy() 

def softmax(x):
    x = np.exp(x)
    total = np.sum(x, axis = -1)
    return x/total[..., np.newaxis] # have to explicity add axis to make broadcasting work

def relu(x):
    return np.maximum(x, 0) 

def illegal_broadcast(x, desired_shape): # allows broadcasting when performing an operation that doesn't allow it (concat)
    dummy = np.zeros(desired_shape, dtype = np.float32)
    return x + dummy

def uniform_initialize(*shape):
    x = (np.random.rand(*shape) - 0.5) / 10
    return x

def xavier_initialize(*shape):
    factor = 1
    for i in shape:
        factor *= i
    limit = np.sqrt(6 / (factor))
    x = (np.random.rand(*shape) - 0.5) * 2 * limit
    return x


def final_layer_initialize(*shape): # removes sqrt to prevent overflow
    factor = 1
    for i in shape:
        factor *= i
    limit = 6 / (factor)
    x = (np.random.rand(*shape) - 0.5) * 2 * limit
    return x


class Embedding:
    def __init__(self, vocab_size, features, positional_features):
        self.vocab_size = vocab_size
        self.features = features
        self.positional_features = positional_features
        self.concat_layer = Concat()
        
    def build(self, dummy_input):
        self.concat_layer.build()
        self.weights = uniform_initialize(self.vocab_size, self.features).astype(np.float32)
        self.positional_embedding = uniform_initialize(dummy_input.shape[-1], self.positional_features).astype(np.float32)

    def call(self, inputs): # inputs is in shape (batch size, tokens)
        embedded_vectors = np.zeros((inputs.shape[0], inputs.shape[1], self.features))
        # pretty sure this works
        index = np.array(inputs)
        sequence = np.array(self.weights[index, :])
        
        positional_sequence = illegal_broadcast(self.positional_embedding, sequence.shape)
        a = self.concat_layer.call((sequence, positional_sequence), axis = -1)
        self.a = a.astype(np.float32)
        return a

    def compute_gradients(self, grads):
        pass


class Dense:
    def __init__(self, neurons, activation = "relu", final_layer = True):
        self.neurons = neurons
        self.activation = activation
        self.final_layer = final_layer
        
    def build(self, dummy_input):
        if not self.final_layer:
            self.weights = xavier_initialize(dummy_input.shape[-1], self.neurons).astype(np.float32)
        else:
            self.weights = final_layer_initialize(dummy_input.shape[-1], self.neurons).astype(np.float32)
        self.biases = np.zeros((1, self.neurons), dtype = np.float32)

    def build_with_shape(self, dummy_shape):
        if not self.final_layer:
            self.weights = xavier_initialize(dummy_shape[-1], self.neurons).astype(np.float32)

        else:
            self.weights = final_layer_initialize(dummy_shape[-1], self.neurons).astype(np.float32)

        self.biases = np.zeros((1, self.neurons), dtype = np.float32)

    def call(self, inputs):
        self.inputs = inputs
        z = np.matmul(inputs, self.weights) + self.biases
        if self.activation == "relu":
            a = relu(z)
        elif self.activation == "softmax":
            a = softmax(z)
        else:
            raise ValueError("Activation not found")
        self.a = a
        return a

    def compute_gradients(self, previous_derivative):
        if self.activation == "relu":
            dadz = np.heaviside(self.a, 0)
            dcdz = np.multiply(dadz, previous_derivative)
            dzdb = 1
            # Use different transpose commands because somehow there's no function that transposes only the 
            # last dimensions 
            if len(self.inputs.shape) == 2: 
                dzdw = self.inputs.T
            elif len(self.inputs.shape) == 3:
                dzdw = np.transpose(self.inputs, (0, 2, 1))
            else:
                raise ValueError("Matrix has unsupported dimensions")

            self.dcdw = np.matmul(previous_derivative, dcdz) / previous_derivative.shape[0]
            self.dcdb = np.sum(dcdz, axis = 0) / dcdz.shape[0]

        else: # For softmax
            # I'm being a little lazy and treading the vector sum as a (value + constant)
            # If training suffers serious issues and everything else works, investigate here
            # Primary issue should be that if I don't account for the vector sum, incorrect options may increase 
            # since there is no penalty keeping it down 
            pass

        dcda = np.matmul(dcdz, self.inputs) / dcda.shape[0]
        return dcda

    def update(self):
        self.weights = self.weights + self.dcdw
        self.biases = self.biases - dcda.shape[0]



class BatchNorm:
    def __init__(self, axis = (0, 1), epsilon = 0.00001):
        self.axis = axis
        self.epsilon = epsilon

    def build(self, dummy_input):
        self.samples = dummy_input.shape[0]
        self.moving_average = np.mean(dummy_input, axis = self.axis) + self.epsilon
        self.moving_std = np.std(dummy_input, axis = self.axis) + self.epsilon

    def call(self, inputs):
        self.inputs = inputs
        a = inputs - self.moving_average / self.moving_std
        self.a = a
        return a

    def compute_gradients(self, previous_derivative):
        dadx = self.a / self.moving_std # Don't calculate the std wrt to inputs
        # Doesn't work unless you're using previous generation inputs as well, and that would just waste memory
        self.dadc = dadx * previous_derivative

        return dadc # Fix for backprop

    def update(self): # 
        current_samples = a.shape[0]
        current_average = np.mean(a, axis = self.axis)
        current_std = np.std(a, axis = self.axis)
        self.moving_average = (self.moving_average * self.samples + current_average * current_samples) / (self.samples + current_samples) # might overflow
        self.moving_std = (self.moving_std * self.samples + current_std * current_samples) / (self.samples + current_samples) # might overflow


class Concat:
    def __init__(self):
        pass 
    def build(self):
        pass
    def call(self, inputs, axis = -1):
        self.inputs = inputs
        self.a = np.concatenate(inputs, axis)
        self.inputs = inputs
        return self.a

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        assert dropout_rate < 1, "unreasonable dropout rate: " + str(dropout_rate) # -1, 2

    def build(self, dummy_input):
        pass

    def call(self, inputs, training = True):
        if training:
            mask = np.random.binomial(1, self.dropout_rate, inputs.shape)
            self.a = np.multiply(inputs, mask) * (1/self.dropout_rate)
            return a
        else:
            return inputs


class Flatten:
    def __init__(self):
        pass 
    def build(self):
        pass
    def call(self, inputs):
        self.inputs = inputs
        self.a = np.reshape(inputs, (inputs.shape[0], -1))
        return self.a

class AttentionHead:
    def __init__(self, neurons):
        self.neurons = neurons
        self.query_layer = Dense(neurons)
        self.key_layer = Dense(neurons)
        self.value_layer = Dense(neurons)

    def build(self, dummy_q, dummy_k, dummy_v = None):
        if dummy_v is None:
            dummy_v = dummy_k
        self.query_layer.build(dummy_q)
        self.key_layer.build(dummy_k)
        self.value_layer.build(dummy_v)

    def call(self, query, key, value = None): # figure out what to store
        self.query = query 
        self.key = key
        if value is None:
            value = key
        self.value = value

        q = self.query_layer.call(query)
        k = self.key_layer.call(key)
        v = self.value_layer.call(value)

        qk = np.matmul(q, np.swapaxes(k, -1, -2)) # transpose the matrix dimensions
        q_shape = q.shape
        k_shape = k.shape
        self.norm_factor = np.sqrt(q_shape[-1] * q_shape[-2] * k_shape[-1] * k_shape[-2])
        qk = qk/self.norm_factor
        qk = softmax(qk)
        a = np.matmul(qk, v)

        return a


class MultiHeadAttention:
    def __init__(self, num_heads, head_dense_shape, final_dense_shape):
        self.head_dense_shape = head_dense_shape
        self.final_dense_shape = final_dense_shape
        self.heads = {}
        self.concat_layers = {}
        self.dense = Dense(final_dense_shape)
        self.num_heads = num_heads
        for i in range(num_heads):
            self.concat_layers[i] = Concat()
            self.heads[i] = AttentionHead(head_dense_shape)

    def build(self, dummy_q, dummy_k, dummy_v = None):
        for i in range(self.num_heads):
            self.concat_layers[i].build()
            self.heads[i].build(dummy_q, dummy_k, dummy_v) 
        self.dense.build_with_shape((dummy_q.shape[-2], self.num_heads * self.head_dense_shape)) # matmuls cancel out the shape

    def call(self, q, k, v = None):
        self.q = q 
        self.k = k
        self.v = v
        head_outputs = []
        for i in range(self.num_heads):
            x = self.heads[i].call(q, k, v)
            if i == 0:
                head_concat = x
            else:
                head_concat = self.concat_layers[i].call((head_concat, x), axis = -1)

        a = self.dense.call(head_concat)
        return a
















