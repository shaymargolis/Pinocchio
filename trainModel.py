import pandas as pd
import numpy as np
from linear_learner import LinearLearner
from logistic_learner import LogisticLearner
from net import NetLearner
from sklearn.model_selection import train_test_split
from addFeatures import features_euclidean_dists
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from getFeatures import eraseUnnamed

def get_data(people, suffix = ""):
    result = pd.DataFrame()

    for person in people:
        #  Read X_values
        true_data = pd.read_csv(person + "/Data/True/All"+suffix+".csv")
        false_data = pd.read_csv(person + "/Data/False/All"+suffix+".csv")

        true_data = eraseUnnamed(true_data)
        false_data = eraseUnnamed(false_data)

        #  Append Y values
        true_data["Truth"] = 1

        false_data["Truth"] = 0

        result = result.append(true_data, sort=False)
        result = result.append(false_data, sort=False)

        print("             (init get_data) Person ", person)
        print("             truth is nan", true_data.isnull().values.any())
        print("             false is nan", false_data.isnull().values.any(), "(done get_data)")

    #  Seperate to X and Y
    Y = result[["Truth"]]
    # X = features_euclidean_dists(result).append([result.drop(["Truth"])], axis = 1)

    X = result.drop(["Truth"], axis = 1)

    return X, Y

def get_model(model_name):
    if model_name == "Linear":
        return LinearLearner()
    if model_name == "Logistic":
        return LogisticLearner()
    if model_name == "Net1":
        return NetLearner("test1", 12425)

    print("Model name ", model_name, "Not found.")
    exit(1)

def trainPersonalizedMethod(train_people, test_people, model_name, suffix = "", label = "", basic = False):
    print("~~~ Running train for ", label, "~~~")
    print("    Getting train data.")
    X_train, Y_train = get_data(train_people, suffix)
    print("    Getting test data.")
    X_test, Y_test = get_data(test_people, suffix)

    if basic and suffix == '_euc':
        ind = createEuclidInd()
        X_train = X_train[ind]
        X_test = X_test[ind]


    print("     The train data:")
    print(X_train)

    print(X_train.columns)

    print(X_train.shape)
    print("     Train is null? ", X_train.isnull().values.any())

    print("     Learning process.")
    model = get_model(model_name)
    model.learn(X_train, Y_train)

    print("    Saving the model.")
    joblib.dump(model.lm, label+".pkl")

    Y_predicted = pd.DataFrame(model.predict(X_train), columns=["Predicted_Truth"])
    Y_predicted = Y_predicted.reset_index(drop=True)

    print(accuracy_score(Y_predicted, Y_train))

def createEuclidInd():
    result = []
    ind = [str(i) for i in range(7)]
    ind += [str(i) for i in range(8, 68)]
    ind += ["left_eye_", "left_iris_", "right_eye_", "right_iris_"]
    for i in range(0, len(ind)):
        for j in range(i+1, len(ind)):
            k = ind[i]
            p = ind[j]
            result.append(k+" DIST FROM "+p)

    return result

def testPersonalizedMethod(label, test_people, suffix = "", basic = False):
    model = joblib.load(label+".pkl")
    X_test, Y_test = get_data(test_people, suffix)


    if basic and suffix == '_euc':
        ind = createEuclidInd()
        X_test = X_test[ind]

    Y_predicted = pd.DataFrame(model.predict(X_test), columns=["Predicted_Truth"])

    Y_test_show = pd.DataFrame(Y_test, columns=["Truth"])

    Y_predicted = Y_predicted.reset_index(drop=True)
    Y_test_show = Y_test_show.reset_index(drop=True)

    Y_test_show = pd.concat([Y_test_show, Y_predicted], axis=1)

    print(Y_test_show)

    feature_importances = pd.DataFrame(model.coef_[0],
                                       index = X_test.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)

    print("~~~ FEATURE IMPORTANCE: " + str(feature_importances))

    print("~~~ TEST SCORE =", int(10000*accuracy_score(Y_predicted, Y_test))/100, "%")
