from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from helper import readResult, save_model_sk
import numpy as np

np.random.seed(1337)
TOLERANCE = 1e-5


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
    save_model_sk(tfidf_vectorizer, './variables/svm_sgd_tfidf_vect.sav')

    print("train_tfidf_shape:", train_words_tfidf.shape)
    print("test_tfidf_shape", test_words_tfidf.shape)
    # print("targets_len:", len(targets_train))

    return {'tfidf': train_words_tfidf, 'target': targets_train}, {'tfidf': test_words_tfidf, 'target': targets_test}


def svm_classify(trains, tests):
    trains, tests = word_transform(trains, tests)
    print("trains[0]:")
    print(trains['tfidf'])
    model = make_pipeline(StandardScaler(with_mean=False), LinearSVC(tol=TOLERANCE, max_iter=2000000))
    model.fit(trains['tfidf'], trains['target'])

    predicted = model.predict(tests['tfidf'])

    print("predicted: ", predicted)
    print("target   : ", tests['target'])

    readResult(tests['target'], predicted, name="SVM", form="JSON")
    return model


def sgd_classify(trains, tests):
    trains, tests = word_transform(trains, tests)

    model = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(max_iter=1000, tol=TOLERANCE))
    model.fit(trains['tfidf'], trains['target'])

    predicted = model.predict(tests['tfidf'])

    readResult(tests['target'], predicted, name="SGD", form="JSON")
    return model
