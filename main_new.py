
import os
import sys
import pandas as pd

import matplotlib.pyplot as plt

from numpy import linalg as LA
import numpy as np

from net import NetLearner

from tqdm import tqdm
from analyzer import Analyzer
from image_source import ImageSource
from sixty_eight import SixtyEightInterpreter
from linear_learner import LinearLearner

from scipy import signal

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def createCouples(df):
    s = len(df.columns.values)

    for i in range(0, s, 2):
        for j in range(i+1, s, 2):
            if i+1 >= s or j+1 >= s:
                continue
            x0, y0, x1, y1 = df[str(i)], df[str(i+1)], df[str(j)], df[str(j+1)]
            df[str(i)+"00"+str(j)] = LA.norm([x0-x1, y0-y1])
            print(100*(i*s+j)/(s**2))

    return df

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
        """self.nl = self.analyze_folder("NL")
        self.nl.to_csv(self.folder + "/nl.csv")
        self.pl = self.analyze_folder("PL")
        self.pl.to_csv(self.folder + "/pl.csv")
        self.nt = self.analyze_folder("NT")
        self.nt.to_csv(self.folder + "/nt.csv")
        self.pt = self.analyze_folder("PT")
        self.pt.to_csv(self.folder + "/pt.csv")"""

        self.truth = self.analyze_folder("Truth")
        self.truth.to_csv(self.folder + "/truth.csv")
        self.false = self.analyze_folder("False")
        self.flase.to_csv(self.folder + "/false.csv")


    """
    Do linear regression on the given data
    splits the data into train&test
    """
    def linear_regression(self):
        #  Create dummies of the Types

        """self.nl = pd.read_csv(self.folder + "/nl.csv")
        self.pl = pd.read_csv(self.folder + "/pl.csv")
        self.nt = pd.read_csv(self.folder + "/nt.csv")
        self.pt = pd.read_csv(self.folder + "/pt.csv")

        self.nl["Type"] = "NL"
        self.pl["Type"] = "PL"
        self.nt["Type"] = "NT"
        self.pt["Type"] = "PT"

        full = self.nl.append([self.pl, self.nt, self.pt])"""


        self.truth = pd.read_csv("AllTruth/AllTruth.csv")
        self.false = pd.read_csv("AllLie/AllLie.csv")

        self.truth["Type"] = "T"
        self.false["Type"] = "F"

        full = self.truth.append([self.false])



        encoder = LabelEncoder()
        encoder.fit(full["Type"])
        encoded_Y = encoder.transform(full["Type"])
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)

        y_data = full["Type"]
        for label in ['Type', 'Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1']:
            if label in full.columns.values:
                full = full.drop([label], axis=1)
        #full = full.drop(columns=['136', '137', '138', '139', '140', '141', '142'])

        print("Nan = ", full.isnull().any())

        """removeList = full.columns.values
        print(removeList)
        full = createCouples(full)
        full = full.drop(columns = removeList)"""

        self.labels = full.columns.values
        #  Create train & test data
        #this.X_train, this.X_test, this.y_train, this.y_test = #train_test_split(np.array(full), np.array(y_data))
        X_train, y_train = full, dummy_y

        print(X_train, "\n", y_train)

        #  Learn the data and check for success
        #self.learner = NetLearner("test2", 137)
        self.learner = LinearLearner()
        self.learner.learn(np.array(X_train), np.array(y_train))

        print(full, "\n\nLearning Done.")

        X_test_truth = pd.read_csv("TruthTest/TruthTest.csv")
        X_test_lie = pd.read_csv("LieTest/LieTest.csv")

        X_test_truth["Type"] = "T"
        X_test_lie["Type"] = "F"

        full_test = X_test_truth.append([X_test_lie])


        encoder = LabelEncoder()
        encoder.fit(full_test["Type"])
        encoded_Y = encoder.transform(full_test["Type"])
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y_test = np_utils.to_categorical(encoded_Y)

        y_test = dummy_y_test#full_test["Type"]
        for label in ['Type', 'Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1']:
            if label in full_test.columns.values:
                full_test = full_test.drop([label], axis=1)

        # self.learner.print_accuracy(full_test, dummy_y_test)

        y_test_predict = pd.DataFrame(self.learner.predict(full_test)).idxmax(axis=1)
        y_test = pd.DataFrame(y_test).idxmax(axis=1)
        #print(y_test, y_test_predict)
        result = np.array(y_test) - np.array(y_test_predict)

        total = len(full_test)
        curr = total - np.count_nonzero(result)

        #  Print the success rate
        print("Total success rate: %.2f%%" % (curr/total * 100))

        importance0 = pd.DataFrame(np.abs(self.learner.lm.coef_[0]),
                                   index = full_test.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

        importance1 = pd.DataFrame(np.abs(self.learner.lm.coef_[1]),
                                   index = full_test.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)


        importance0 = np.array(importance0) / 2
        importance1 = np.array(importance1) / 2
        print(importance0, '\n', importance1)
        with open("importance.txt", "w") as file:
            file.write("///////Importance 0///////\n")
            file.write(str(importance0))
            file.write("\n\n///////Importance 1///////\n")
            file.write(str(importance1))
        """

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

    """
    Predicts the type of the video based on
    the data and the learner.
    """
    def predict_type(self, title, data, person = "", state = "", index = 0):
        #  Create pattern of result
        #result = pd.DataFrame([0, 0, 0, 0], index=["NL", "NT", "PL", "PT"])
        result = pd.DataFrame([0, 0], index=["Truth", "False"])

        #  Get y based on learner
        y_predict = pd.DataFrame(self.learner.predict(data)).idxmax(axis=1)

        #  Get the most frequent one
        freq = y_predict.value_counts()

        print(freq)

        for i in range(min(len(freq), 4)):
            result.iloc[freq.index[i], 0] = freq.iloc[i]

        print(result)
        print(result[0].tolist())

        colors = {'/PL/': 'r', '/NL/': 'b', '/PT/': 'g', '/NT/': 'darkorange', '/Test/': 'mediumslateblue', '/Real_Test/':'mediumslateblue'}
        plt.figure(dpi=150)
        plt.grid()
        plt.title("Distribution of Frame Predictions\nin video #"+str(index)+" of "+person+", " + state[1:3])
        plt.xlabel("Predicted type")
        plt.ylabel("# Of frames")
        plt.bar(["NL", "NT", "PL", "PT"], result[0].tolist(), color = colors[state])

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
res2 = pd.DataFrame(result)
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

        if file_list[i][-4:] not in [".mov", ".mp4"]:
            continue

        if analyze:
            print(file_list[i])
            source = ImageSource(root + "/" + file_list[i])
            pg = tqdm(total=source.get_length())
            sixty = SixtyEightInterpreter()
            anal = Analyzer(source, sixty, False, pg)
            result = anal.start()
            pg.close()
            res2 = pd.DataFrame(result)

            res2.to_csv(root +"/"+ file_list[i] + ".csv")
        res2 = pd.read_csv(root +"/"+ file_list[i] + ".csv")
"""
        blink = res2.loc[:, '134']

        blink_num = analyze_blink(blink)
        truth.append(blink_num)

        print("["+file_list[i]+"] Blink num ", blink_num)

    truth = np.array(list(filter(lambda x:x!=(-1,-1), truth)))

    print("["+title+"] Average blink num", np.average(truth), np.std(truth))

    return truth"""

show = False
name = "Topaz_Enbar"

def concatCsvs(folder, name):
    file_list = os.listdir(folder)
    result = pd.DataFrame()
    for file in file_list:

        if not file[-4:] == ".csv":
            continue
        path = folder+"/"+file
        if result.empty:
            result = pd.read_csv(path)
        else:
            result = pd.concat([result, pd.read_csv(path)], axis = 0, sort = False)

    result.to_csv(folder +"/"+ name + ".csv")
    return result

#NL = analyze_folder("NL", "Videos/"+name+"/NL", show)
#PL = analyze_folder("PL", "Videos/"+name+"/PL", show)
#PT_blink = analyze_folder("PT-blink", "Videos/"+name+"/PT-blink", show)
#NT = analyze_folder("NT", "Videos/"+name+"/NT", show)

""""pa = PersonAnalyzer("Videos/Topaz_Enbar")
pa.linear_regression()

for file in os.listdir("Videos/Topaz_Enbar/Test"):
    if not file[-4:] == ".csv":
        continue

    if file in ["test.csv", "Importance.csv"]:
        continue

    X_test = pd.read_csv("Videos/Topaz_Enbar/Test/"+file)

    if 'Unnamed: 0' in X_test.columns.values:
        X_test = X_test.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0.1' in X_test.columns.values:
        X_test = X_test.drop(columns=['Unnamed: 0.1'])

    removeList = X_test.columns.values
    print(removeList)
    X_test = createCouples(X_test)
    X_test = X_test.drop(columns=removeList)

    y_predict = pa.learner.predict(np.array(X_test))
    plt.figure(dpi = 250)
    plt.plot(y_predict, 'ro')
    plt.title(file)
    plt.savefig("Videos/Topaz_Enbar/Test/"+file+".eps")

feature_importances = pd.DataFrame(np.abs(pa.learner.lm.coef_),
                                   index = pa.labels,
                                   columns=['importance']).sort_values('importance', ascending=False)
feature_importances.to_csv("Videos/Topaz_Enbar/Test/Importance.csv")"""

"""for file in os.listdir("Videos/Topaz_Enbar/Test"):
    if not file[-4:] == ".csv":
        continue"""


def eraseNan(person, subfolders):
    for state in subfolders:
        df = pd.read_csv(state+"/"+state+".csv")
        for col in [134, 135, 136, 137, 140, 141, 142]:
            lst = list(df[str(col)])

            start = 0
            while np.isnan(lst[start]):
                    start += 1

            for i in range(start):
                lst[i] = lst[start]

            for i in range(start, len(lst)):
                if np.isnan(lst[i]):
                    lst[i] = lst[i-1]
            df[str(col)] = lst
        df.to_csv(state+"/"+state+".csv")

def eraseUnnamed(person, subfolders):
    for folder in subfolders:
        df = pd.read_csv(folder+"/"+folder+".csv")
        for label in ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1']:
            if label in df.columns.values:
                df = df.drop(columns = [label])
        df.to_csv(folder+"/"+folder+".csv")

pa = PersonAnalyzer("")
pa.linear_regression()

#eraseNan("", ["AllTruth", "AllLie", "AllTest"])

"""person = ""
subfolders = ["LieTest", "TruthTest"]
for folder in subfolders:
    concatCsvs(folder, folder)

eraseNan(person, subfolders)
eraseUnnamed(person, subfolders)"""
"""NT = analyze_folder("REAL-NT", "Videos/"+name+"/Real-NT", show)
NT = analyze_folder("REAL-PT", "Videos/"+name+"/Real-PT", show)
NT = analyze_folder("REAL-PL", "Videos/"+name+"/Real-PL", show)
NT = analyze_folder("REAL-NL", "Videos/"+name+"/Real-NL", show)"""
