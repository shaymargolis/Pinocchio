import pandas as pd
import numpy as np
from linear_learner import LinearLearner
from logistic_learner import LogisticLearner
from net import NetLearner
from sklearn.model_selection import train_test_split
from addFeatures import features_euclidean_dists
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def get_data(people):
    result = pd.DataFrame()

    for person in people:
        #  Read X_values
        true_data = pd.read_csv(person + "/Data/True/All.csv")
        false_data = pd.read_csv(person + "/Data/False/All.csv")

        #  Append Y values
        true_data["Truth"] = 1

        false_data["Truth"] = 0

        result = result.append(true_data, sort=False)
        result = result.append(false_data, sort=False)

        print("Person ", person)
        print("truth is nan", true_data.isnull().values.any())
        print("false is nan", false_data.isnull().values.any())

    #  Seperate to X and Y
    Y = result[["Truth"]]

    ind = [str(i+1) + x for i in range(1) for x in ['x','y']]
    ind += [str(i+1) + x for i in range(8, 68) for x in ['x','y']]
    # X = features_euclidean_dists(result).append([result.drop(["Truth"])], axis = 1)
    X = result.drop(["Truth"], axis = 1)

    return X, Y

def get_model(model_name):
    if model_name == "Linear":
        return LinearLearner()
    if model_name == "Logistic":
        return LogisticLearner()
    if model_name == "Net1":
        return NetLearner("test1", 2145)

    print("Model name ", model_name, "Not found.")
    exit(1)

def trainPersonalizedMethod(train_people, test_people, model_name, label = ""):
    X_train, Y_train = get_data(train_people)
    X_test, Y_test = get_data(test_people)

    print(X_train)

    print(X_train.columns)

    print(X_train.shape)
    print("Train is null", X_train.isnull().values.any())

    model = get_model(model_name)


    model.learn(X_train, Y_train)

    Y_predicted = pd.DataFrame(model.predict(X_test), columns=["P_Truth"])

    Y_test_show = pd.DataFrame(Y_test, columns=["Truth"])

    Y_predicted = Y_predicted.reset_index(drop=True)
    Y_test_show = Y_test_show.reset_index(drop=True)

    Y_test_show = pd.concat([Y_test_show, Y_predicted], axis=1)

    print(Y_test_show)

    # print(model.model.get_weights())

    print(accuracy_score(Y_predicted, Y_test))

    joblib.dump(model.lm, label+".pkl")


def testPersonalizedMethod(label, test_people):
    X_test, Y_test = get_data(test_people)
    model = joblib.load(label+".pkl")
    feature_importances = pd.DataFrame(model.lm.coef_[0],
                                       index = X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    print("~~~ FEATURE IMPORTANCE: " + str(feature_importances))
    model.print_accuracy(X_test, Y_test)
    print(accuracy_score(Y_predicted, Y_test))
