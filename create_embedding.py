import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

dataset = pd.read_csv("./NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt", sep="\t", header=None, names=["Word", "Valence", "Arousal", "Dominance"])

embeddings = model.encode(dataset["Word"].tolist())

np.save("embeddings.npy", embeddings)
























































# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import statsmodels.api as sm
# import pickle
# import time

# embedding_model = SentenceTransformer('all-mpnet-base-v2')

# def train_and_evaluate(data_path, save_path):
#     start = time.time()
#     data = pd.read_csv(data_path)
#     print("Data loaded")
#     print(f"Time taken: {time.time() - start} seconds")

#     embeddings = embedding_model.encode(data["post"].tolist(), convert_to_numpy=True)
#     print("Embeddings computed")
#     print(f"Time taken: {time.time() - start} seconds")

#     train_size = int(0.8 * len(embeddings))
#     X_train = embeddings[:train_size]
#     y_train = data.iloc[:train_size, 2]
#     print("Training data prepared")
#     print(f"Time taken: {time.time() - start} seconds")

#     X_val = embeddings[train_size:]
#     y_val = data.iloc[train_size:, 2]
#     print("Validation data prepared")
#     print(f"Time taken: {time.time() - start} seconds")

#     X_train_const = sm.add_constant(X_train)
#     model = sm.GLM(y_train, X_train_const, family=sm.families.Binomial())
#     result = model.fit()
#     print("Model trained")
#     print(f"Time taken: {time.time() - start} seconds")

#     X_val_const = sm.add_constant(X_val)
#     predictions = result.predict(X_val_const)
#     MSE = ((predictions - y_val) ** 2).mean()
#     RMSE = np.sqrt(MSE)
#     print(f"{data_path} - MSE: {MSE}, RMSE: {RMSE}")
#     print(f"Time taken: {time.time() - start} seconds")

#     with open(save_path, "wb") as file:
#         pickle.dump(result, file)
#     print("Model saved")
#     print(f"Time taken: {time.time() - start} seconds")

# populations = ['M', 'F']
# dimensions = ['v', 'a', 'd']

# for pop in populations:
#     for dim in dimensions:
#         data_path = f"./data.gender.vad.scores/data.{pop}.{dim}.csv"
#         save_path = f"./pickle/new_binom.model.{pop}.{dim}.pkl"
#         train_and_evaluate(data_path, save_path)

# print("All models retrained and saved")
