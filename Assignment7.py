# Dataset Source: http://ai.stanford.edu/~amaas/data/sentiment/
# Implementation idea: http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/

import nltk.classify.util as util
from nltk.classify import NaiveBayesClassifier
import os


def word_feats(words):
    return dict([(word, True) for word in words])


def get_words(file_loc, filename):
    f_ptr = open(file_loc + "/" + filename, encoding="utf-8")
    content = str(f_ptr.read())
    f_ptr.close()

    return content.split()


def get_loc(type, emotion):
    return "data/%s/%s" % (type, emotion)


def get_feats(type, emotion):
    file_loc = get_loc(type, emotion)
    return [(word_feats(get_words(file_loc, f)), emotion) for f in os.listdir(file_loc)]


def get_features(type):
    neg_features = get_feats(type, 'neg')
    pos_features = get_feats(type, 'pos')

    return neg_features + pos_features

training_data = get_features("train")
testing_data = get_features("test")

print('train on %d instances, test on %d instances' % (len(training_data), len(testing_data)))

classifier = NaiveBayesClassifier.train(training_data)
accuracy = round(util.accuracy(classifier, testing_data) * 100, 4)

print('accuracy: ', accuracy)
classifier.show_most_informative_features()
