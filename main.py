
import os
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
        self.nl["Type"] = "NL"
        self.pl["Type"] = "PL"
        self.nt["Type"] = "NT"
        self.pt["Type"] = "PT"

        full = nl.append([pl, nt, pt])

        y_data = pd.get_dummies(full.loc[:, 'Type'])
        full = full.drop(['Unnamed: 0', 'Type'], axis=1)

        #  Create train & test data
        X_train, X_test, y_train, y_test = train_test_split(np.array(full), np.array(y_data))

        #  Learn the data and check for success
        learner = LinearLearner()
        learner.learn(X_train, y_train)

        #  Get the predicted output for X_test and compare to
        #  the expected output
        y_test_predict = pd.DataFrame(learner.predict(X_test)).idxmax(axis=1)
        y_test = pd.DataFrame(y_test).idxmax(axis=1)

        #  Count number of failures
        result = np.array(y_test) - np.array(y_test_predict)

        total = len(X_test)
        curr = total - np.count_nonzero(result)

        #  Print the success rate
        print("Total success rate %.2f%%" % (curr/total * 100))

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

# pa = PersonAnalyzer("Videos/roy_amir")
#  pa.analyze_data()

pa = PersonAnalyzer("Videos/roy_amir")
pa.analyze_data()
pa.linear_regression()

"""source = ImageSource("Videos/roy amir telling the truth.MOV")
orig_interpreter = SixtyEightInterpreter()
analyzer = Analyzer(source, orig_interpreter)
result_t = np.array(analyzer.start())

result_ty = np.ones(len(result_t)) * 1

data = pd.DataFrame(result_t)
data.to_csv('truth.csv')

source = ImageSource("Videos/roy_amir_tru.MOV")
orig_interpreter = SixtyEightInterpreter()
analyzer = Analyzer(source, orig_interpreter)
result_f = np.array(analyzer.start())

data = pd.DataFrame(result_f)
data.to_csv('false.csv')

result_fy = np.zeros(len(result_f)) * 0

result = np.concatenate(result_t, result_f)
result_y = np.concatenate(result_ty, result_fy)

learner = LinearLearner()
print("A")
learner.learn(result, result_y)
print("B")"""

#  data = pd.DataFrame(result)
#  print(data)
