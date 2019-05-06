from trainModel import train_test_person
from logistic_learner import LogisticLearner
from sklearn.externals import joblib
import pandas as pd
import numpy as np

def analyze(X_train, Y_train, X_test, Y_test):
    print(X_train.shape)
    print(Y_train.shape)

    lm = LogisticLearner()
    lm.learn(X_train, Y_train)

    lies_detected, lie_exist = 0, 0
    truth_detected, truth_exist = 0, 0

    for i in range(len(X_test)):
        file_x, x = X_test[i]
        file_y, y = Y_test[i]


        try:
            real_val = y["Truth"][0] #  If rthe question should be FALSE or TRUE
        except:
            print("Error in file " + file_x)
            traceback.print_exc()
            continue

        Y_predicted = pd.DataFrame(lm.predict(x), columns=["Predicted_True"])

        Y_test_show = pd.DataFrame(y, columns=["Truth"])

        Y_predicted = Y_predicted.reset_index(drop=True)
        Y_test_show = Y_test_show.reset_index(drop=True)

        Y_test_show = pd.concat([Y_test_show, Y_predicted], axis=1)

        a = Y_predicted["Predicted_True"].value_counts()

        if real_val == 0:
            lie_exist += 1
        if real_val == 1:
            truth_exist += 1

        print("~~~ Test question that is %d [%s] ~~~" % (real_val, file_x))
        # print(Y_test_show)
        lie, truth = 0, 0
        try:
            lie = a[0]
        except:
            pass
        try:
            truth = a[1]
        except:
            pass
        print("Number of lie: " + str(lie) + "/" + str(lie+truth))
        print("Number of truth: " + str(truth) + "/" + str(lie+truth))

        if lie >= truth and real_val == 0:
            lies_detected += 1
        if truth >= lie and real_val == 1:
            truth_detected += 1

    print("~~~ Summary ~~~")
    print("Lies detected: %d/%d" % (lies_detected, lie_exist))
    print("Truth detected: %d/%d" % (truth_detected, truth_exist))

    return lm
