# import lib
from document import *
from cnn import cnn_zhang, cnn_improved
from lstm import lstm_zhang, lstm_improved
from svm_sgd import svm_classify, sgd_classify
import numpy as np
from helper import save_vocab, save_model_sk, save_model_keras

# dataset initialize
trainlist,devlist, testlist = initData(1,7)
max_len = 0;
# for i in trainlist[0]:
#     if(max_len<len(i)):
#         max_len=len(i)
# print("max length", max_len)

week_count = len(trainlist)
print("Weeks: ", week_count)

documents = []
for i in range(week_count):
    documents += trainlist[i] + devlist[i] + testlist[i]
DF(documents)
vocab = getVocabulary(documents)
print("Vocab: ", len(vocab))
save_vocab(vocab)

X_train, y_train = formatK(trainlist[0], vocab)
X_test, y_test = formatK(testlist[0], vocab)
print()
#print("---------SVM---------")
svm_model = svm_classify(trainlist[0], devlist[0])
# #print("---------SGD---------")
sgd_model = sgd_classify(trainlist[0], testlist[0])
#print("---------CNN---------")
cnn_model = cnn_zhang(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
# #print("---------CNN+--------")
# cnn_improved(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))

# #print("---------LSTM--------")
lstm_model = lstm_zhang(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))
# #print("--------LSTM+--------")
lstm_plus_model = lstm_improved(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), len(vocab))

#saving the models to file
save_model_sk(svm_model, './variables/svm_model.sav')
save_model_sk(sgd_model, './variables/sgd_model.sav')
save_model_keras(cnn_model, './variables/cnn_model.h5')
save_model_keras(lstm_model, './variables/lstm_model.h5')
save_model_keras(lstm_plus_model, './variables/lstm+_model.h5')
