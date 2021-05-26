import twint
from document import unigram, isEnglishWord, formatK_tweet
from helper import load_vocab, load_model_sk, load_model_keras, word_transform, load_model_keras_custom
from lstm import root_mean_squared_logarithmic_error
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np


def tweet_scrap(search):
    data = []
    twint_conf = twint.Config()
    tweets = ""
    twint_conf.Search = search
    twint_conf.verified = True
    twint_conf.Lang = "en"
    twint_conf.Links = "exclude"
    twint_conf.Popular_tweets = True
    twint_conf.Lowercase = True
    twint_conf.Show_hashtags = False
    twint_conf.Limit = 20
    twint_conf.Hide_output = True
    twint_conf.Store_object = True
    twint_conf.Store_object_tweets_list = data
    twint.run.Search(twint_conf)

    for tweet in data:
        for word in tweet.tweet.split():
            if isEnglishWord(word):
                tweets += " " + word
    tweet_word = unigram(tweets)
    return tweet_word


def predict(tweet_word):
    vocab = load_vocab()
    input_data_keras = formatK_tweet(tweet_word, vocab)
    for i in range(128):
        if (len(input_data_keras) < 128):
            input_data_keras.append(0)
    input_data_sk = word_transform(tweet_word)

    cnn = load_model_keras('./variables/cnn_model.tf')
    lstm = load_model_keras('./variables/lstm_model.tf')
    lstm_improved = load_model_keras_custom('./variables/lstm+_model.tf',
                                            {"peephole_lstm_cells": tfa.rnn.PeepholeLSTMCell(32),
                                             "root_mean_squared_logarithmic_error": root_mean_squared_logarithmic_error})
    svm = load_model_sk('./variables/svm_model.sav')
    sgd = load_model_sk('./variables/sgd_model.sav')

    # print("Prediction:")
    # print("CNN: ", cnn.predict([input_data_keras[:128]]))
    # print("LSTM: ", lstm.predict([input_data_keras[:128]]))
    # print("LSTM+: ", lstm_improved.predict([input_data_keras[:128]]))
    # print("SVM: ", svm.predict(input_data_sk))
    # print("SGD: ", sgd.predict(input_data_sk))
    return {"CNN": cnn.predict([input_data_keras[:128]])[0][0], "LSTM": lstm.predict([input_data_keras])[0][0],
            "LSTM+": lstm_improved.predict([input_data_keras])[0][0], "SVM": svm.predict(input_data_sk)[0],
            "SGD": sgd.predict(input_data_sk)[0]}


def main():
    tweet_search = input("Tweet keyword: ")

    tweets = tweet_scrap(tweet_search)
    prediction = predict(tweets)
    print(prediction)

    plt_keys = prediction.keys()
    plt_values = prediction.values()
    x_pos = np.arange(len(plt_values))
    plt_colors = []
    for i in plt_values:
        if i >= 0.5:
            plt_colors.append("green")
        else:
            plt_colors.append("red")

    plt.title("DDOS attack prediction\nKeyword: " + tweet_search)
    plt.bar(x_pos, plt_values, color=plt_colors)
    plt.xticks(x_pos, plt_keys)
    plt.ylabel("Probabilty")
    plt.xlabel("Model")
    plt.show()


if __name__ == "__main__":
    main()
