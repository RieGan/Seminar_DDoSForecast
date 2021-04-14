from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from helper import readResult
import numpy as np

np.random.seed(1337)

def word_transform(documents_train, documents_test):
    words_train = []
    targets_train = []
    for document in documents_train:
        words_train.append(' '.join(map(str, list(document.words.keys()))))
        targets_train.append(document.polarity)
    # print(words)

    words_test = []
    targets_test = []
    for document in documents_test:
        words_test.append(' '.join(map(str, list(document.words.keys()))))
        targets_test.append(document.polarity)

    # count_vect = CountVectorizer()
    # words_counts = count_vect.fit_transform(words)
    #
    # print("count_shape:", words_counts.shape)
    #
    # tfidf_transformer = TfidfTransformer()
    # words_tfidf = tfidf_transformer.fit_transform(words_counts)


    tfidf_vectorizer = TfidfVectorizer()
    train_words_tfidf = tfidf_vectorizer.fit_transform(words_train)
    test_words_tfidf = tfidf_vectorizer.transform(words_test)

    print("train_tfidf_shape:", train_words_tfidf.shape)
    print("test_tfidf_shape", test_words_tfidf.shape)
    #print("targets_len:", len(targets_train))

    return {'tfidf': train_words_tfidf, 'target': targets_train}, {'tfidf': test_words_tfidf, 'target': targets_test}



def svm_classify(trains, tests):

    trains, tests = word_transform(trains, tests)

    model = LinearSVC()
    model.fit(trains['tfidf'], trains['target'])

    predicted = model.predict(tests['tfidf'])

    # print("predicted: ", predicted)
    # print("target   : ", tests['target'])

    return readResult(tests['target'], predicted)
