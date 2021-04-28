import math
import json
from sklearn.metrics import average_precision_score


def readResult(y_test, results, form=None, name=None):
    index = 0
    auc = average_precision_score(y_test, results)
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

    if (tp + fp) == 0 and (tn + fn) != 0:
        precisionP = "err:div by zero"
        precisionN = tn / (tn + fn)
    elif (tp + fp) != 0 and (tn + fn) == 0:
        precisionP = tp / (tp + fp)
        precisionN = "err:div by zero"
    elif (tp + fp) == 0 and (tn + fn) == 0:
        precisionP = "err:div by zero"
        precisionN = "err:div by zero"
    else:
        precisionP = tp / (tp + fp)
        precisionN = tn / (tn + fn)

    if (tp + fn) == 0 and (tn + fp) != 0:
        recallP = "err:div by zero"
        recallN = tn / (tn + fp)
    elif (tp + fn) != 0 and (tn + fp) == 0:
        recallP = tp / (tp + fn)
        recallN = "err:div by zero"
    elif (tp + fn) == 0 and (tn + fp) == 0:
        recallP = "err:div by zero"
        recallN = "err:div by zero"
    else:
        recallP = tp / (tp + fn)
        recallN = tn / (tn + fp)

    if(recallP == "err:div by zero" or recallN == "err:div by zero"):
        gmean = "err:div by zero"
    else:
        gmean = math.sqrt(recallP * recallN)

    if(precisionP == "err:div by zero" or recallP =="err:div by zero" or (precisionP + recallP) == 0):
        f_p = "err:div by zero"
    else:
        f_p = 2 * precisionP * recallP / (precisionP + recallP)

    if(precisionN == "err:div by zero" or recallN=="err:div by zero" or (precisionN + recallN) ==0):
        f_n = "err:div by zero"
    else:
        f_n = 2 * precisionN * recallN / (precisionN + recallN)

    if form == "JSON":
        obj = {
            "Name": name,
            "G-mean": gmean,
            "PrecisionPositive": precisionP,
            "PrecisionNegative": precisionN,
            "RecallPositive": recallP,
            "RecallNegative": recallN,
            "F1-ScorePositive": f_p,
            "F1-ScoreNegative": f_n,
            "Accuracy": acc,
            "AUC": auc
        }

        print(json.dumps(obj))

    elif form is None:
        print("--------------------------")
        print(" ModelName: ", name)
        print("--------------------------")
        print("Gmean:", gmean)
        print("recallP:", recallP)
        print("recallN:", recallN)
        print("precP:", precisionP)
        print("precN:", precisionN)
        print("fP:", f_p)
        print("fN:", f_n)
        print("acc:", acc)
        print("AUC:", auc)
