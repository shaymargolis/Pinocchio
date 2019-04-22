import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
import subprocess
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

def runOnAllVideosFiles(persons, options, function, input = None):
    for person in persons:
        for option in options:
            dir = "Videos/" + person + "/" + option
            file_list = os.listdir(dir)
            for file in file_list:
                filepath = dir + "/" + file
                function((filepath, input))

def cutVideo(tuple):
    filepath, length = tuple
    with VideoFileClip(filepath) as video:
        new = video.subclip(0, length)
        new.write_videofile(filepath[:-4]+"_cut.mp4", audio_codec='libmp3lame')

def convertMov(tuple):
    filepath, extension = tuple
    savepath = "Audio/" + filepath[7:-4] + extension
    command = "ffmpeg -i "+filepath+" "+savepath
    subprocess.call(command, shell=True)


def getFFT(filepath, s = 2048, savepath = ""):
        rate, x = wavfile.read(filepath)
        y = x[:, 1]
        pxx, freqs, bins, _ = plt.specgram(y, NFFT=s, Fs=rate, noverlap=0,
                                           cmap=plt.cm.binary, sides='onesided',
                                           window=signal.blackmanharris(s),
                                           scale_by_freq=True,
                                           mode='magnitude')

        if not savepath == "":
            plt.figure(dpi = 250)
            plt.plot(freqs, 20 * np.log10(np.mean(pxx, axis=1)), color = 'darkviolet')
            plt.savefig(savepath)
        return 20 * np.log10(np.mean(pxx, axis=1))


#getFFT("Audio/Oded Kaplan/PT/PT-01.wav", savepath = "Audio/Oded Kaplan/PT/PT-01.eps")


persons = ["Oded_Kaplan"]
options = ["PT", "NT", "NL", "PL"]
runOnAllVideosFiles(persons, options, convertMov, ".mp4")
#cutVideo(("Videos/Oded_Kaplan/PT/PT-21.mov", 0.7))
