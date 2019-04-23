import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
import subprocess
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

from scipy.interpolate import interp1d
from scipy.signal      import argrelextrema

def eac(sig, winsize=512, rate=44100):
	"""Return the dominant frequency in a signal."""
	s = np.reshape(sig[:len(sig)//winsize*winsize], (-1, winsize))
	s = np.multiply(s, np.hanning(winsize))
	f = np.fft.fft(s)
	p = (f.real**2 + f.imag**2)**(1/3)
	f = np.fft.rfft(p).real
	q = f.sum(0)/s.shape[1]
	q[q < 0] = 0
	intpf = interp1d(np.arange(winsize//2), q[:winsize//2])
	intp = intpf(np.linspace(0, winsize//2-1, winsize))
	qs = q[:winsize//2] - intp[:winsize//2]
	qs[qs < 0] = 0
	return qs

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
    savepath = "Videos/" + filepath[7:-4] + extension
    command = "ffmpeg -i "+filepath+" -vf scale=1080:720 "+savepath
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
        return np.fft.fftfreq(s, 1/rate), 20 * np.log10(np.mean(pxx, axis=1))


#getFFT("Audio/Oded Kaplan/PT/PT-01.wav", savepath = "Audio/Oded Kaplan/PT/PT-01.eps")

def getHighestFreq(filepath, save = False):
    freq, y = getFFT(filepath)
    freq = freq[:int(len(freq)/2)]
    x = np.linspace(0, len(y)/rate, len(y))
    if save:
        plt.figure(dpi = 250)
        plt.plot(freq, y[:-1], color='pink')
        plt.savefig("Audio/Oded_Kaplan/PT/u2-FFT.eps")

    return freq[y.argmax()]

persons = ["Oded_Kaplan"]
options = ["PT", "NT", "NL", "PL"]
#runOnAllVideosFiles(persons, options, convertMov, ".avi")
#cutVideo(("Videos/Oded_Kaplan/PT/PT-21.mov", 0.7))
