
import os
import pandas as pd

import numpy as np

from ProgressBar import ProgressBar
from analyzer import Analyzer
from image_source import ImageSource
from sixty_eight import SixtyEightInterpreter
from linear_learner import LinearLearner

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
        nl = self.analyze_folder("NL")
        pl = self.analyze_folder("PL")
        nt = self.analyze_folder("NT")
        pt = self.analyze_folder("PT")

        nl.to_csv(self.folder + "/nl.csv")
        pl.to_csv(self.folder + "/pl.csv")
        nt.to_csv(self.folder + "/nt.csv")
        pt.to_csv(self.folder + "/pt.csv")


    """
    Do linear regression on the given data
    splits the data into train&test
    """
    def linear_regression(self):
        pass

    """
    Analyzes (Gets features of) a specific folder
    inside the main folder given in the constructor
    for all the videos in the folder
    """
    def analyze_folder(self, name):
        #  Create result DataFrame
        result = pd.DataFrame()

        #  Make root folder path
        #  and get file list in location
        root = self.folder + "/" + name
        file_list = os.listdir(root)

        #  Create an ProgressBar
        pg = ProgressBar(len(file_list))

        for file in file_list:
            #  Update the progressBar
            pg.update()

            #  Analyze the file
            path_file = root + "/" + file

            source = ImageSource(path_file)
            orig_interpreter = SixtyEightInterpreter()
            analyzer = Analyzer(source, orig_interpreter, False)
            result_t = np.array(analyzer.start())

            data = pd.DataFrame(result_t)
            result = result.append(data)

        return result

pa = PersonAnalyzer("roy_amir")
pa.analyze_data()

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
