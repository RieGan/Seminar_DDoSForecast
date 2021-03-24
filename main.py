# import lib
from document import *
# from cnn import cnn_zhang, cnn_improved
from lstm import lstm_zhang, lstm_improved
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
print("trainlist")
for i in trainlist:
    print(len(i))
print("devlist")
for i in devlist:
    print(len(i))
print("testlist")
for i in testlist:
    print(len(i))
print(trainlist[0][0])
print(devlist[0][0])
X_train,y_train=formatK(trainlist[0],vocab)
X_test,y_test=formatK(devlist[0],vocab)
print(X_train[0])
print(y_train)
print(X_test[0])

# cnn_zhang(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
# cnn_improved(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
lstm_zhang(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
lstm_improved(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
