import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

class Serialization:
    @staticmethod
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)


embeddings = np.load("./embeddings.npy")

dataset = pd.read_csv("./NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt", sep="\t", header=None, names=["Word", "Valence", "Arousal", "Dominance"])

def train_regressor(X, y):
    X_const = sm.add_constant(X) 
    regressor = sm.GLM(y, X_const, family=sm.families.Binomial()).fit()
    return regressor

regressor_a = train_regressor(embeddings, dataset["Arousal"])
regressor_v = train_regressor(embeddings, dataset["Valence"])
regressor_d = train_regressor(embeddings, dataset["Dominance"])

serializer = Serialization()
serializer.save_obj(regressor_a, './pickle/retrained_new_binom.model.a')
serializer.save_obj(regressor_v, './pickle/retrained_new_binom.model.v')
serializer.save_obj(regressor_d, './pickle/retrained_new_binom.model.d')




