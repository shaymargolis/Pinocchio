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
from tqdm import tqdm
import traceback
import os

def filter2_data(data):
    #  Gather only wanted features
    ind = [(0, 21), (17, 21), (19, 21), (19, 37), (38, 40), ("left_eye_", "left_iris_"),
           (22, 26), (22, 24), (24, 44), (16, 24), (43, 47), ("right_eye_", "right_iris_"),
           (48, 54), (52, 56), (62, 66)] # Chin`

    res = []
    for x,y in ind:
        res.append(str(x)+" DIST FROM "+str(y))

    #  Now take all features that is important
    ind = [a + x for a in res for x in ["_avg", "_std", "/dt"]]

    real_res = []
    a = data.columns
    for r in ind:
        if r in a:
            real_res.append(r)

    data = data.loc[:, real_res]

    return data

def read_files(person, state, files):
    x, y = [], []

    print("Reading", state, "directory.")
    pg = tqdm(total=len(files))

    for file in files:
        pg.update()

        try:
            data = pd.read_csv("Videos/New/" + person + "/Data/" + state + "/" + file)
            data = eraseUnnamed(data)

            data = filter2_data(data)

            if state == "True":
                data["Truth"] = 1
            if state == "False":
                data["Truth"] = 0
            if state == "Test":
                if "lie" in file:
                    data["Truth"] = 0
                if "truth" in file:
                    data["Truth"] = 1

            y.append((file, pd.DataFrame(data["Truth"], columns=["Truth"])))
            x.append((file, data.drop(["Truth"], axis=1)))

        except:
            print("FILE FAILED: "+file+"\n")
            traceback.print_exc()
            pass

    pg.close()

    return x, y

def get_person_split(person, suffix = ""):
    true = []
    false = []

    for state in ["True", "False"]:
        files = os.listdir("Videos/New/" + person + "/Data/" + state)

        files = list(filter(lambda x: ("All" not in x) and x.endswith(suffix + ".csv"), files))

        if state == "True":
            true = files
        if state == "False":
            false = files

    return true, false

def concatenate_true_false(X1, X2, Y1, Y2):
    X = pd.DataFrame()
    Y = pd.DataFrame()

    T_X, T_Y = X1+X2, Y1+Y2

    for i in range(len(T_X)):
        file_x, x = T_X[i]
        file_y, y = T_Y[i]

        X = X.append(x, sort=False)
        Y = Y.append(y, sort=False)

    return X, Y

def train_test_person(person, suffix = "", test_size=0.25):
    true, false = get_person_split(person, suffix)
    files_train1, files_test1, _, _ = train_test_split(true, true, test_size=test_size)
    files_train2, files_test2, _, _ = train_test_split(false, false, test_size=test_size)

    print("X_train1", files_train1)
    print("\n\nX_test1", files_test1)
    print("\n\nX_train2", files_train2)
    print("\n\nX_test2", files_test2)

    X_train1, Y_train1 = read_files(person, "True", files_train1)
    X_test1, Y_test1 = read_files(person, "True", files_test1)
    X_train2, Y_train2 = read_files(person, "False", files_train2)
    X_test2, Y_test2 = read_files(person, "False", files_test2)

    # Concatenate train and set
    X_train, Y_train = concatenate_true_false(X_train1, X_train2, Y_train1, Y_train2)
    X_test, Y_test = X_test1+X_test2, Y_test1+Y_test2

    return X_train, Y_train, X_test, Y_test

def multpl(ind):
    res = []
    for i in range(0, len(ind)):
        for j in range(0, len(ind)):
            k = str(ind[i])
            p = str(ind[j])

            res.append(k+" DIST FROM "+p)

    return res

def filter_data(data):
    res = set()

    ind = ["left_eye_", "left_iris_"]
    ind += [36, 37, 38, 39] # Left eye
    res.update(multpl(ind))

    ind = ["right_eye_", "right_iris_"]
    ind += [42, 44, 46, 45] # Right eye
    res.update(multpl(ind))

    ind = [17, 19, 21]
    ind += [36, 37, 38, 39]  # Left eye
    res.update(multpl(ind))

    ind =  [22, 24, 26] # Right eyebrow
    ind += [42, 44, 46, 45] # Right eye
    res.update(multpl(ind))

    ind = [0, 3, 8, 12, 16] # Chin

    res = [a + x for a in res for x in ["_avg", "_std"]]

    real_res = []
    a = data.columns
    for r in res:
        if r in a:
            real_res.append(r)

    return data[real_res]

def get_data(people, suffix = ""):
    result = pd.DataFrame()

    for person in people:
        #  Read X_values
        true_data = pd.read_csv("Videos/New/" +person + "/Data/True/All"+suffix+".csv")
        false_data = pd.read_csv("Videos/New/" +person + "/Data/False/All"+suffix+".csv")

        true_data = eraseUnnamed(true_data)
        false_data = eraseUnnamed(false_data)

        true_data = filter2_data(true_data)
        false_data = filter2_data(false_data)

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
        return NetLearner("test1", 120)

    print("Model name ", model_name, "Not found.")
    exit(1)

def trainPersonalizedMethod(train_people, test_people, model_name, suffix = "", label = ""):
    print("~~~ Running train for ", label, "~~~")
    print("    Getting train data.")
    X_train, Y_train = get_data(train_people, suffix)
    print("    Getting test data.")
    X_test, Y_test = get_data(test_people, suffix)


    print("     The train data:")
    print(X_train)

    print(X_train.columns)

    print(X_train.shape)
    print("     Train is null? ", X_train.isnull().values.any())

    print("     Learning process.")
    model = get_model(model_name)
    model.learn(X_train, Y_train)

    print("    Saving the model.")
    # joblib.dump(model.lm, label+".pkl")


def testPersonalizedMethod(label, test_people, suffix = ""):
    model = joblib.load(label+".pkl")
    X_test, Y_test = get_data(test_people, suffix)

    Y_predicted = pd.DataFrame(model.predict(X_test), columns=["Predicted_True"])

    print(Y_predicted["Predicted_True"].value_counts())

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
