import sys
import csv
import pickle
import numpy as np

import statsmodels.api as sm
from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from scipy.stats import ranksums
from statistics import mean, stdev
from math import sqrt


class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
        serialization of an object
        :param obj: object to serialize
        :param name: file name to store the object
        """
        with open('pickle/' + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
        de-serialization of an object
        :param name: file name to load the object from
        """
        with open('pickle/' + name + '.pkl', 'rb') as fout:
            return pickle.load(fout)
        # end with
    # end def

# end class


def get_sentences(text):
    sentences = list()
    for s in sent_tokenize(text):
        if len(s.split()) > MIN_SENTENCE_LEN: sentences.append(s)
    # end for
    return sentences
# end def


def infer_emotion_value(embeddings, regressor):
    predictions = regressor.predict(embeddings)
    assert(len(predictions) == len(embeddings))
    return predictions

# end def


def vad_analysis_for_population(population, type):
    data = Serialization.load_obj('week2comments.' + population)
    regressor = Serialization.load_obj('binom.model.' + type)

    model = SentenceTransformer('bert-large-nli-mean-tokens')

    filename = dirname + 'data.'+population+'.' + type + '.csv'
    with open(filename, 'w') as fout:
        csv_writer = csv.writer(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['week', 'post'] + [type.capitalize(), 'std('+type.capitalize()+')'])
        for week in range(len(data)):
            for i, post in enumerate(data[week]):
                if i % 100 == 0:
                    print('processing week:', week, 'post:', i)
                    sys.stdout.flush()
                # end if

                sentences = get_sentences(post)
                if len(sentences) == 0: continue  # no sentences long enough
                values = infer_emotion_value(model.encode(sentences), regressor)  # predict dimension score
                csv_writer.writerow([str(week), post, np.mean(values), np.std(values)])
            # end for
        # end for
    # end with

# end def


def test_differences(dimension):
    scores_m = read_data(dirname + 'data.M.' + dimension + '.csv')
    scores_f = read_data(dirname + 'data.F.' + dimension + '.csv')

    cohens_d = (mean(scores_m) - mean(scores_f)) / (sqrt((stdev(scores_m) ** 2 + stdev(scores_f) ** 2) / 2))
    print(dimension, len(scores_m), len(scores_f), format.format(mean(scores_m)), format.format(mean(scores_f)),
          format.format(cohens_d), '\t', ranksums(scores_m, scores_f)[1])

# end def


def read_data(filename):
    scores = list()
    with open(filename, 'r', encoding='utf8') as fin:
        csv_reader = csv.reader(fin, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = next(csv_reader)
        for line in csv_reader: scores.append(float(line[2]))
    # end with

    return scores

# end def


dimension = 'v'
format = '{:.6f}'
MIN_SENTENCE_LEN = 5
dirname = 'data.gender.vad.scores/'

if __name__ == '__main__':

    vad_analysis_for_population('F', dimension)
    vad_analysis_for_population('M', dimension)

    test_differences(dimension)


# end if
