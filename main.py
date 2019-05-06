"""from getFeatures import get_features

people = ["Topaz_Enbar"]
get_features(people, True)"""

import trainModel
from logistic_learner import LogisticLearner
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import test
import os

"""people = ["Topaz_Enbar"]

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

suffix = "_euc"

from tqdm import tqdm

def filter_files(files):
    return list(filter(lambda x: ("All" not in x) and x.endswith("_euc.csv"), files))

for person in people:
    print("~~~ Filtering person " + person + " ~~~")
    vals = ["True", "False"]
    if person in ["Topaz_Enbar"]:
        vals += ["Test"]

    for a in vals:

        files =  filter_files(os.listdir("Videos/New/" + person + "/Data/"+a))

        pg = tqdm(total=len(files))

        for file in files:
            test_data = pd.read_csv("Videos/New/"+person + "/Data/"+a+"/" +  file)

            test_data = filter2_data(test_data)

            test_data.to_csv("Videos/New/"+person + "/Data/"+a+"/" + file)

            pg.update()

        pg.close()"""


"""trainModel.trainPersonalizedMethod(
    ["Nadav_Finkel", "Roy_Amir", "Topaz_Enbar", "Oded_Kaplan", "Raziel_Gartzman"],
    ["Raveh_Shulman"],
    "Net1",
    label="raveh",
    suffix="_euc"
)

trainModel.testPersonalizedMethod("raveh", ["Raveh_Shulman"], suffix = "_euc")
"""
"""X, Y = get_data(["Raveh_Shulman"])

model = joblib.load("saved_model.pkl")
result = model.score(X, Y)
print(result)

print(model.coef_)
print(X.columns)

feature_importances = pd.DataFrame(np.abs(lm.lm.coef_[0]),
                               index = X_train.columns,
                               columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)
5
model.print_accuracy(X_test, Y_test)"""

#  UNIVERSAL

trainModel.trainPersonalizedMethod(
    ["Nadav_Finkel", "Roy_Amir", "Topaz_Enbar", "Oded_Kaplan", "Raziel_Gartzman"],
    ["Raveh_Shulman"],
    "Net1",
    label="raveh",
    suffix="_euc"
)

trainModel.testPersonalizedMethod("raveh", ["Raveh_Shulman"], suffix = "_euc")

#  FILTER FILE OLD
#


person = "Nadav_Finkel"
X_train, Y_train, X_test, Y_test = trainModel.train_test_person(person, suffix = "_euc", test_size=0.3)

lm = test.analyze(X_train, Y_train, X_test, Y_test)

feature_importances = pd.DataFrame(lm.lm.coef_[0],
                               index = X_train.columns,
                               columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)

#  FILTER FILES OLD + TODAY

def filter_files(files):
    return list(filter(lambda x: ("All" not in x) and x.endswith("_euc.csv"), files))

person = "Topaz_Enbar"
X_train1, Y_train1 = trainModel.read_files(person, "True", filter_files(os.listdir("Videos/New/" + person + "/Data/True")))
X_train2, Y_train2 = trainModel.read_files(person, "False", filter_files(os.listdir("Videos/New/" + person + "/Data/False")))
X_train, Y_train = trainModel.concatenate_true_false(X_train1, X_train2, Y_train1, Y_train2)

X_test, Y_test = trainModel.read_files(person, "Test", filter_files(os.listdir("Videos/New/" + person + "/Data/Test")))

lm = test.analyze(X_train, Y_train, X_test, Y_test)

feature_importances = pd.DataFrame(lm.lm.coef_[0],
                               index = X_train.columns,
                               columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)
