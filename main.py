
import os
import sys
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from analyzer import Analyzer
from image_source import ImageSource
from sixty_eight import SixtyEightInterpreter
from linear_learner import LinearLearner

from scipy import signal

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
        self.nl.to_csv(self.folder + "/nl.csv")
        self.pl = self.analyze_folder("PL")
        self.pl.to_csv(self.folder + "/pl.csv")
        self.nt = self.analyze_folder("NT")
        self.nt.to_csv(self.folder + "/nt.csv")
        self.pt = self.analyze_folder("PT")
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
        self.learner = LinearLearner()
        self.learner.learn(X_train, y_train)

        # Merge PL+NL and PT+NT
        def group(n):
            if n in [0, 1]:
                return 0
            return 1

        #  Get the predicted output for X_test and compare to
        #  the expected output
        y_test_predict = pd.DataFrame(self.learner.predict(X_test)).idxmax(axis=1).apply(group)
        y_test = pd.DataFrame(y_test).idxmax(axis=1).apply(group)

        #  Count number of failures
        result = np.array(y_test) - np.array(y_test_predict)

        total = len(X_test)
        curr = total - np.count_nonzero(result)

        #  Print the success rate
        print("Total success rate: %.2f%%" % (curr/total * 100))

    """
    Predicts the type of the video based on
    the data and the learner.
    """
    def predict_type(self, title, data):
        #  Create pattern of result
        result = pd.DataFrame([0, 0, 0, 0], index=["NL", "NT", "PL", "PT"])

        #  Get y based on learner
        y_predict = pd.DataFrame(self.learner.predict(data)).idxmax(axis=1)

        #  Get the most frequent one
        freq = y_predict.value_counts()

        print(freq)

        for i in range(min(len(freq), 4)):
            result.iloc[freq.index[i], 0] = freq.iloc[i]

        print(result)
        print(result[0].tolist())

        plt.figure(dpi=150)
        plt.grid()
        plt.title("Distribution of frames in video for " + title)
        plt.xlabel("Predicted type")
        plt.ylabel("# Of frames")
        plt.bar(["NL", "NT", "PL", "PT"], result[0].tolist())

        return

    """
    Opens a specific video type and tries to
    guess using self.learner the type of the
    video.
    """
    def analyze_video(self, root, file_list, index, display = False, pg = None):
        #  Get the path file
        path_file = root + "/" + file_list[index]
        source = ImageSource(path_file)

        #  Analyze data
        orig_interpreter = SixtyEightInterpreter()
        analyzer = Analyzer(source, orig_interpreter, False, pg)
        result_t = np.array(analyzer.start())

        data = pd.DataFrame(result_t)

        return data

    """
    Returns the length in frames of a folder.
    """
    def get_length_of_folder(self, root, file_list):
        #  Create ImageSource from every
        #  file and count the total length
        images = []
        length_sum = 0
        for file in file_list:
            path_file = root + "/" + file
            source = ImageSource(path_file)
            images.append(source)

            length_sum += source.get_length()

        return length_sum

    """
    Analyzes (Gets features of) a specific folder
    inside the main folder given in the constructor
    for all the videos in the folder
    """
    def analyze_folder(self, name, display = False):
        #  Print message
        print("Starting analyze of " + name + ":")

        #  Create result DatafgFrame
        result = pd.DataFrame()

        #  Make root folder path
        #  and get file list in location
        root = self.folder + "/" + name
        file_list = os.listdir(root)

        #  Create ImageResource list
        length_sum = self.get_length_of_folder(root, file_list)

        #  Create an ProgressBar
        pg = tqdm(total=length_sum)

        for i in range(len(file_list)):
            #  Analyze the video
            data = self.analyze_video(root, file_list, i, display, pg)
            result = result.append(data)

        pg.close()

        return result

"""source = ImageSource("Videos/kaplan/NT/NT-21.mov")
sixty = SixtyEightInterpreter()
anal = Analyzer(source, sixty, True)
result = anal.start()
res2 = pd.DataFrame(result)"""

"""# person = sys.argv[1]
pa = PersonAnalyzer("Videos/" + "kaplan")
pa.linear_regression()

root = "Videos/kaplan/PT"
file_list = os.listdir(root)

for i in range(len(file_list)):
    print(file_list[i])

    result = pa.analyze_video(root, file_list, i)

    if (len(result) == 0):
        continue

    pa.predict_type(file_list[i], result)

    plt.savefig("Figs/" + file_list[i] + "_distr.png")

source = ImageSource("Videos/kaplan/NL/kaplan_lies_no_1.mp4")
sixty = SixtyEightInterpreter()
anal = Analyzer(source, sixty, True)
result = anal.start()
res1 = pd.DataFrame(result)

res1.to_csv("res1.csv")

source = ImageSource("Videos/kaplan/NT/NT-21.mov")
sixty = SixtyEightInterpreter()
anal = Analyzer(source, sixty, True)
result = anal.start()
res2 = pd.DataFrame(result)

res2.to_csv("res2.csv")

plt.figure()
plt.plot(res1[136], "r-")
plt.plot(res2[136], "b-")
plt.show()
"""

from scipy.signal import find_peaks


def closest_to_down(data, abs_min, step):
    i = abs_min
    while i > 0 and i < len(data)-1 and data[i] < 0.05:
        i += step

    return i

def analyze_blink(data):
    try:
        data = np.array(data)
        peaks, _ = find_peaks(data)

        abs_min = np.array(data).argmin()

        closest_left = closest_to_down(data, abs_min, -1)
        closest_right = closest_to_down(data, abs_min, 1)

        blink_start = (data[closest_left] - data[abs_min]) / (abs_min - closest_left)
        blink_end = (data[closest_right] - data[abs_min]) / (closest_right - closest_left)

        filt = signal.savgol_filter(data, 7, 4)

        a = np.linspace(-0.02, -0.02, len(data))

        plt.figure()
        plt.title("INTERPRETATION OF BLINK")
        plt.plot(data, "r.")
        plt.plot(data, "r-", label="raw data")
        plt.plot(filt, "b-", label="smooth data")
        plt.plot(a, "y-", label="threshold")
        plt.plot(np.gradient(data), "g-", label="gradient on raw data")
        plt.plot(np.gradient(filt), "y-", label="gradient on smoothed data")
        plt.plot(closest_left, [data[closest_left]], "b.")
        plt.plot(closest_right, [data[closest_right]], "y.")
        plt.legend()
        plt.show()

        filtgrad = np.gradient(filt)
        peaks = filtgrad[ np.where( filtgrad < -0.02 ) ]

        return len(peaks)

    except Exception as e:
        print(e)
        plt.figure()
        plt.title("BAD INTERPRETATION OF BLINK")
        plt.plot(data, "r-")
        plt.show()
        return -1, -1

def analyze_folder(title, root, analyze=True):
    file_list = os.listdir(root)

    truth = []

    for i in range(len(file_list)):
        if analyze:
            print(file_list[i])
            source = ImageSource(root + "/" + file_list[i])
            pg = tqdm(total=source.get_length())
            sixty = SixtyEightInterpreter()
            anal = Analyzer(source, sixty, False, pg)
            result = anal.start()
            pg.close()
            res2 = pd.DataFrame(result)

            res2.to_csv("Data/" + root + "/" + file_list[i] + ".csv")
        res2 = pd.read_csv("Data/" + root + "/" + file_list[i] + ".csv")

        blink = res2.loc[:, '134']

        blink_num = analyze_blink(blink)
        truth.append(blink_num)

        print("["+file_list[i]+"] Blink num ", blink_num)

    truth = np.array(list(filter(lambda x:x!=(-1,-1), truth)))

    print("["+title+"] Average blink num", np.average(truth), np.std(truth))

    return truth

show = False
name = 'kaplan'

#NL = analyze_folder("NL", "Videos/"+name+"/NL", show)
#PL = analyze_folder("PL", "Videos/"+name+"/PL", show)
#PT_blink = analyze_folder("PT-blink", "Videos/"+name+"/PT-blink", show)
#NT = analyze_folder("NT", "Videos/"+name+"/NT", show)
analyze_folder("kaplan", "Videos/inter/kaplan", False)
"""NT = analyze_folder("REAL-NT", "Videos/"+name+"/Real-NT", show)
NT = analyze_folder("REAL-PT", "Videos/"+name+"/Real-PT", show)
NT = analyze_folder("REAL-PL", "Videos/"+name+"/Real-PL", show)
NT = analyze_folder("REAL-NL", "Videos/"+name+"/Real-NL", show)"""
