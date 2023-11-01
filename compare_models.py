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
with open("coefficients_comparison.txt", "w") as file:
    for e in types:
        original_regressor = Serialization.load_obj('./pickle/binom.model.' + e)
        new_regressor = Serialization.load_obj('./pickle/retrained_new_binom.model.' + e)
        
        file.write(f"Type: {e}\n")
        file.write("Original Coefficients:\n")
        file.write(str(original_regressor.params))
        file.write("\n\nNew Coefficients:\n")
        file.write(str(new_regressor.params))
        file.write("\n\n" + "="*50 + "\n\n")
