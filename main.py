
import os
import sys
import pandas as pd

import numpy as np

from tqdm import tqdm
from analyzer import Analyzer
from image_source import ImageSource
from sixty_eight import SixtyEightInterpreter
from linear_learner import LinearLearner

from sklearn.model_selection import train_test_split

class PersonAnalyzer:
    """
    Create new PersonAnalyzer for
    specific person in folder
    (Folder consists of NL, PL, NT, PT folders)
    """
    def __init__(self, folder):
        self.folder = folder

    """
    Calls analyze_folder for all 4 folders
    of raw data
    """
    def analyze_data(self):
        self.nl = self.analyze_folder("NL")
        self.pl = self.analyze_folder("PL")
        self.nt = self.analyze_folder("NT")
        self.pt = self.analyze_folder("PT")

        self.nl.to_csv(self.folder + "/nl.csv")
        self.pl.to_csv(self.folder + "/pl.csv")
        self.nt.to_csv(self.folder + "/nt.csv")
        self.pt.to_csv(self.folder + "/pt.csv")


    """
    Do linear regression on the given data
    splits the data into train&test
    """
    def linear_regression(self):
        #  Create dummies of the Types

        self.nl = pd.read_csv(self.folder + "/nl.csv")
        self.pl = pd.read_csv(self.folder + "/pl.csv")
        self.nt = pd.read_csv(self.folder + "/nt.csv")
        self.pt = pd.read_csv(self.folder + "/pt.csv")

        self.nl["Type"] = "NL"
        self.pl["Type"] = "PL"
        self.nt["Type"] = "NT"
        self.pt["Type"] = "PT"

        full = self.nl.append([self.pl, self.nt, self.pt])

        y_data = pd.get_dummies(full.loc[:, 'Type'])
        full = full.drop(['Unnamed: 0', 'Type'], axis=1)

        #  Create train & test data
        X_train, X_test, y_train, y_test = train_test_split(np.array(full), np.array(y_data))

        #  Learn the data and check for success
        learner = LinearLearner()
        learner.learn(X_train, y_train)

        # Merge PL+NL and PT+NT
        def group(n):
            if n in [0, 1]:
                return 0
            return 1

        #  Get the predicted output for X_test and compare to
        #  the expected output
        y_test_predict = pd.DataFrame(learner.predict(X_test)).idxmax(axis=1).apply(group)
        y_test = pd.DataFrame(y_test).idxmax(axis=1).apply(group)

        #  Count number of failures
        result = np.array(y_test) - np.array(y_test_predict)

        total = len(X_test)
        curr = total - np.count_nonzero(result)

        #  Print the success rate
        print("Total success rate: %.2f%%" % (curr/total * 100))

    """
    Analyzes (Gets features of) a specific folder
    inside the main folder given in the constructor
    for all the videos in the folder
    """
    def analyze_folder(self, name):
        #  Print message
        print("Starting analyze of " + name + ":")

        #  Create result DataFrame
        result = pd.DataFrame()

        #  Make root folder path
        #  and get file list in location
        root = self.folder + "/" + name
        file_list = os.listdir(root)

        #  Create ImageResource list
        images = []
        length_sum = 0
        for file in file_list:
            path_file = root + "/" + file
            source = ImageSource(path_file)
            images.append(source)

            length_sum += source.get_length()

        #  Create an ProgressBar
        pg = tqdm(total=length_sum)

        for source in images:
            #  Analayze the video
            orig_interpreter = SixtyEightInterpreter()
            analyzer = Analyzer(source, orig_interpreter, False, pg)
            result_t = np.array(analyzer.start())

            data = pd.DataFrame(result_t)
            result = result.append(data)

        pg.close()

        return result

person = sys.argv[1]
pa = PersonAnalyzer("Videos/" + person)
# pa.analyze_data()
pa.linear_regression()
