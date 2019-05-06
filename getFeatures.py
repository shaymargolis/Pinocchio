import os
import pandas as pd
import numpy as np
import traceback

from sixtyEight import get_sixtyeight_feature
from addFeatures import features_avg
from addFeatures import features_euclidean_dists
from addFeatures import features_std

def get_features(people, basic_override = False, euclid = False):
    """
    This function creates dataframes (vectors) as csv files in the Data folders
    for each of the people in the given list (for every video in their folder).
    """

    suffix = ""

    for person in people:
        for state in ["True", "False"]:

            files = os.listdir(person + "/Video/" + state)

            for file in files:

                if not file[-4:] in [".mov", ".mp4"]:
                    continue

                try:
                    path = {"Video": person + "/Video/" + state + "/" + file, "Data": person + "/Data/" + state + "/" + file[:-4] + ".csv"}

                    # os.system('clear')

                    print("\n\n~~~ PERSON: "+person+" ||| STATE: "+state+" ||| FILE: "+file+" ~~~\n")

                    if basic_override:
                        sixtyeight_dots = get_sixtyeight_feature(path["Video"])
                        sixtyeight_dots.to_csv(path["Data"])
                        sixtyeight_dots = eraseNan(sixtyeight_dots)

                    else:
                        sixtyeight_dots = pd.read_csv(path["Data"])

                    if euclid:
                        sixtyeight_dots = features_euclidean_dists(sixtyeight_dots)
                        suffix = "_euc"

                    print("Done Euclading. Number of columns:", sixtyeight_dots.columns.values.shape)

                    ind = sixtyeight_dots.columns.values.copy()
                    for i in range(len(ind)):
                        ind[i] += "/dt"
                    dt = pd.DataFrame(np.gradient(np.array(sixtyeight_dots), axis = 0), columns = ind)

                    ind = sixtyeight_dots.columns.values.copy()
                    for i in range(len(ind)):
                        ind[i] += "_dfa"
                    dist_from_avg = sixtyeight_dots.subtract(sixtyeight_dots.mean(axis = 0), axis = 1)
                    dist_from_avg = pd.DataFrame(np.array(dist_from_avg), columns = ind)

                    avgs = features_avg(sixtyeight_dots.copy())

                    stds = features_std(sixtyeight_dots.copy())

                    #euclidean_dists = features_euclidean_dists(sixtyeight_dots)

                    result = pd.concat([sixtyeight_dots, dt, dist_from_avg, avgs, stds], axis = 1)
                    result.to_csv(person + "/Data/" + state + "/" + file[:-4] + suffix+".csv")

                except Exception:
                    print("FILE FAILED: "+file+"\n")
                    traceback.print_exc()


def eraseNan(features):
    for col in [134, 135, 136, 137, 140, 141, 142]:
        lst = list(features.iloc[:, col])

        start = 0
        while np.isnan(lst[start]):
            if start < len(lst):
                start += 1
            else:
                return features

        for i in range(start):
            lst[i] = lst[start]

        for i in range(start, len(lst)):
            if np.isnan(lst[i]):
                lst[i] = lst[i-1]

        features.iloc[:, col] = lst
    #df.to_csv(state+"/"+state+".csv")
    return eraseUnnamed(features)

def eraseUnnamed(features):
    for label in features.columns.values:
        if label.startswith("Unnamed:"):
            features = features.drop([label], axis = 1)
    return features

def concatenateData(people, suffix = ""):
    """
    This function concatenates all data in a persons data folder to one main dataframe csv file.
    """
    for person in people:
        for state in ["True", "False"]:

            print("~~~ PERSON: "+person+" ||| STATE: "+state+" ~~~")

            files = os.listdir(person + "/Data/" + state)
            dataframes = []

            for file in files:
                if not file[(-4-len(suffix)):] == suffix+".csv":
                    continue
                if file == "All"+suffix+".csv":
                    continue
                print("~~~          CONCATENATING FILE: "+file+" ~~~")
                dataframes.append(pd.read_csv(person + "/Data/" + state + "/" + file))

            result = pd.concat(dataframes, axis = 0, sort = True)
            result.to_csv(person + "/Data/" + state + "/All"+suffix+".csv")
            print("~~~ FINAL DIMENSIONS: "+str(result.shape)+" ~~~")
            #return result
