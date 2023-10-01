import numpy as np
import pickle
import time
import layers
import sys

class Model:
    def __init__(self):
        self.EngEmbed = layers.Embedding(10001, 5, 5)
        self.GermEmbed = layers.Embedding(10001, 5, 5)

        self.EngHead = layers.MultiHeadAttention(3, 19, 23)
        self.GermHead = layers.MultiHeadAttention(3, 19, 23)
        self.MixedHead = layers.MultiHeadAttention(3, 29, 31)

        self.EngLinear = layers.Dense(7)
        self.MixedLinear = layers.Dense(17)
        self.OutputLinear = layers.Dense(10001, activation = "softmax")

        self.EngNorm1 = layers.BatchNorm()
        self.EngNorm2 = layers.BatchNorm()
        self.GermNorm1 = layers.BatchNorm()
        self.MixedNorm1 = layers.BatchNorm()
        self.MixedNorm2 = layers.BatchNorm()

        self.EngConc1 = layers.Concat()
        self.EngConc2 = layers.Concat()

        self.GermConc1 = layers.Concat()
        self.MixedConc1 = layers.Concat()
        self.MixedConc2 = layers.Concat()

        self.Flatten = layers.Flatten()



    def build(self, x_eng, x_germ):
        # Encoder
        self.EngEmbed.build(x_eng)
        embed_eng = self.EngEmbed.call(x_eng)

        self.EngHead.build(embed_eng, embed_eng)
        x_eng = self.EngHead.call(embed_eng, embed_eng)

        self.EngConc1.build()
        x_eng = self.EngConc1.call((x_eng, embed_eng))

        self.EngNorm1.build(x_eng)
        x_eng_norm = self.EngNorm1.call((x_eng))

        self.EngLinear.build(x_eng_norm)
        x_eng = self.EngLinear.call(x_eng_norm)

        self.EngConc2.build()
        x_eng = self.EngConc2.call((x_eng_norm, x_eng))
        #Encoder end, decoder start
        self.GermEmbed.build(x_germ)
        embed_germ = self.GermEmbed.call(x_germ)

        self.GermHead.build(embed_germ, embed_germ)
        x_germ = self.GermHead.call(embed_germ, embed_germ)



        self.GermHead.build(embed_germ, embed_germ)
        x_germ = self.GermHead.call(embed_germ, embed_germ)

        self.GermConc1.build()
        x_germ = self.GermConc1.call((x_germ, embed_germ))

        self.GermNorm1.build(x_germ)
        x_germ = self.GermNorm1.call(x_germ)
        # Begin mixing language inputs here
        self.MixedHead.build(x_eng, x_eng, x_germ)
        x = self.MixedHead.call(x_eng, x_eng, x_germ)

        self.MixedConc1.build()
        x = self.MixedConc1.call((x, x_germ))

        self.MixedNorm1.build(x)
        x_norm = self.MixedNorm1.call(x)

        self.MixedLinear.build(x_norm)
        x = self.MixedLinear.call(x_norm)

        self.MixedConc2.build()
        x = self.MixedConc2.call((x, x_norm))

        self.Flatten.build()
        x = self.Flatten.call(x)

        print(x.shape)
        self.OutputLinear.build(x)
        print("HERE", np.max(self.OutputLinear.weights), self.OutputLinear.weights.shape)
        output = self.OutputLinear.call(x)
        print(output.shape, output.dtype)








    def call(self, x_eng, x_germ):
        embed_eng = self.EngEmbed.call(x_eng)
        x_eng = self.EngHead.call(embed_eng, embed_eng)
        x_eng = self.EngConc1.call((x_eng, embed_eng))
        x_eng_norm = self.EngNorm1.call((x_eng))
        x_eng = self.EngLinear.call(x_eng_norm)
        x_eng = self.EngConc2.call((x_eng_norm, x_eng))



X_train_eng = np.load(f"X_train_eng2.npy").astype(np.int32)



X_train_germ = np.load(f"X_train_germ2.npy").astype(np.int32)
Y_train = np.load(f"Y_train2.npy").astype(np.int32)

transformer = Model()
transformer.build(X_train_eng[:300, :], X_train_germ[:300, :])





