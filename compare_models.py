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

types = ["v", "a", "d"]
for e in types:
    original_regressor = Serialization.load_obj('./pickle/binom.model.' + e)
    new_regressor = Serialization.load_obj('./pickle/retrained_new_binom.model.' + e)
    print("Original Coefficients:", original_regressor.params)
    print("New Coefficients:", new_regressor.params)
