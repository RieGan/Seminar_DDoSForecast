import math
from sklearn.metrics import average_precision_score

def readResult(y_test, results):
    index = 0
    p = n = tp = tn = fp = fn = 0
    for prob in results:
        if prob > 0.5:
            predLabel = 1
        else:
            predLabel = 0
        if y_test[index] > 0:
            p += 1
            if predLabel > 0:
                tp += 1
            else:
                fn += 1
        else:
            n += 1
            if predLabel == 0:
                tn += 1
            else:
                fp += 1
        index += 1
    print("TruePositive:", tp)
    print("TrueNegative:", tn)
    print("falsePositive:", fp)
    print("falseNegative:", fn)
    acc = (tp + tn) / (p + n)
    precisionP = tp / (tp + fp)
    precisionN = tn / (tn + fn)
    recallP = tp / (tp + fn)
    recallN = tn / (tn + fp)
    gmean = math.sqrt(recallP * recallN)
    f_p = 2 * precisionP * recallP / (precisionP + recallP)
    f_n = 2 * precisionN * recallN / (precisionN + recallN)
    print("Gmean:", gmean)
    print("recallP:", recallP)
    print("reacallN:", recallN)
    print("precP:", precisionP)
    print("precN:", precisionN)
    print("fP:", f_p)
    print("fN:", f_n)
    print("acc:", acc)
    print("AUC:",average_precision_score(y_test, results))


