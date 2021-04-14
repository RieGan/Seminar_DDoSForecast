# import lib
from document import *
from cnn import cnn_zhang, cnn_improved
from lstm import lstm_zhang, lstm_improved
from svm import svm_classify
import numpy as np

# dataset initialize
trainlist, devlist, testlist = initData_multi()

week_count = len(trainlist)
print("Weeks: ", week_count)

documents = []
for i in range(week_count):
    documents += trainlist[i] + devlist[i] + testlist[i]
DF(documents)
vocab = getVocabulary(documents)
print("Vocab: ", len(vocab))

X_train, y_train = formatK(devlist[0], vocab)
X_test, y_test = formatK(testlist[0], vocab)

print()
print("---------SVM---------")
svm_classify(devlist[0], testlist[0])
print("---------CNN---------")
cnn_zhang(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
print("---------CNN+--------")
cnn_improved(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
print("---------LSTM--------")
lstm_zhang(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
print("--------LSTM+--------")
lstm_improved(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
