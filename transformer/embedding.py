import numpy as np
import pickle
import time

# matrix of word embeddings, use indexing to find the proper embedding


class embedding:
    def __init__(self, vocab_size, features):
        self.vocab_size = vocab_size
        self.features = features
        self.weights = np.random.rand(vocab_size, features).astype(np.float32)
    def build(self):
        pass
    def call(self, text): # text is in shape (batch size, tokens)
        embedded_vectors = np.zeros((text.shape[0], text.shape[1], self.features))
        #for # use numpy advanced indexing
        index = np.array(text)
        sequence = np.array(self.weights[index, :])
        return sequence

x = np.zeros((32, 100), dtype = np.int32)
embed = embedding(100, 64)
out = embed.call(x)
print(out.shape)
