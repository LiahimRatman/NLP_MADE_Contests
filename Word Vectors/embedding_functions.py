import numpy as np
import math


def text_to_bow(text, tokens):
    """ convert text string to an array of token counts. Use bow_vocabulary. """
    #<YOUR CODE>
    new_tokens = {}
    for token in tokens:
        new_tokens[token] = 0
    for token in text.split():
        if token in new_tokens:
            new_tokens[token] += 1
    return np.array(list(new_tokens.values()), 'float32')


def splitter(row):
    """ Splits clean row by spaces"""
    row_ = []
    for symbol in row:
        if symbol.isalpha() or symbol == ' ':
            row_.append(symbol)
    return ''.join(row_).lower().split()


def computeReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all
    the unique words in the review and whose values are their
    corresponding tf.
    """
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1
    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict


def computeCountDict(tfDict):
    """ Returns a dictionary whose keys are all the unique words in
    the dataset and whose values count the number of reviews in which
    the word appears.
    """
    countDict = {}
    for review in tfDict:
        for word in review:
            if word in countDict:
                countDict[word] += 1
            else:
                countDict[word] = 1
    return countDict


def computeIDFDict(data_train_tfidf, countDict):
    """ Returns a dictionary whose keys are all the unique words in the
    dataset and whose values are their corresponding idf.
    """
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(data_train_tfidf) / countDict[word])
    return idfDict


def computeReviewTFIDFDict(reviewTFDict, idfDict):
    """ Returns a dictionary whose keys are all the unique words in the
    review and whose values are their corresponding tfidf.
    """
    reviewTFIDFDict = {}
    for word in reviewTFDict:
        reviewTFIDFDict[word] = reviewTFDict[word] * idfDict.get(word, 0)
    return reviewTFIDFDict


def computeTFIDFVector(review, wordDict):
    """ Returns tfidf vector for a review"""
    tfidfVector = [0.0] * len(wordDict)

    for i, word in enumerate(wordDict):
        if word in review:
            tfidfVector[i] = review[word]
    return tfidfVector


def get_phrase_embedding(phrase, model):
    """ Returns phrase embedding for model"""
    vector = np.zeros([model.vector_size], dtype='float32')
    phrase_tokenized = splitter(phrase)
    phrase_vectors = [model[x] for x in phrase_tokenized if x in model.vocab.keys()]
    if len(phrase_vectors) != 0:
        vector = np.mean(phrase_vectors, axis=0)

    return vector
