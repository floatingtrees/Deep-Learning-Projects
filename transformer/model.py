import numpy as np
import pickle
import time
import layers

class Model:
    def __init__(self):
        self.EngEmbed = layers.Embedding(101, 5, 5)
        self.GermEmbed = layers.Embedding(101, 5, 5)
        self.EngHead = layers.MultiHeadAttention(3, 19, 23)
        self.GermHead = layers.MultiHeadAttention(3, 19, 23)
        self.MixedHead = layers.MultiHeadAttention(3, 29, 31)
        self.EngLinear = layers.Dense(7)
        self.MixedLinear = layers.Dense(17)
        self.EngNorm1 = layers.BatchNorm()
        self.EngNorm2 = layers.BatchNorm()
        self.GermNorm1 = layers.BatchNorm()
        self.MixedNorm1 = layers.BatchNorm()
        self.MixedNorm2 = layers.BatchNorm()

    def build()






