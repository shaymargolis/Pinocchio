import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
import subprocess
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

from scipy.interpolate import interp1d
from scipy.signal	  import argrelextrema

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


"""def runOnAllVideosFiles(persons, options, function, input = None):
	res = []
	for person in persons:
		res1 = []
		for option in options:
			res2 = []
			dir =  "Videos/" + person + "/" + option
			file_list = os.listdir(dir)
			for file in file_list:
				filepath = dir + "/" + file
				res2.append(function((filepath, input)))
			res1.append(res2)
		res.append(res1)
	return res"""

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
	return np.fft.fftfreq(s, 1/rate), 20 * np.log10(np.mean(pxx, axis=1))


#getFFT("Audio/Oded Kaplan/PT/PT-01.wav", savepath = "Audio/Oded Kaplan/PT/PT-01.eps")

def getHighestFreq(filepath, save = False):
	freq, y = getFFT(filepath)
	freq = freq[:int(len(freq)/2)]
	if save:
		plt.figure(dpi = 250)
		plt.plot(freq, y[:-1], color='pink')
		plt.savefig(filepath[:-4]+"_FFT.eps")

	return freq[y.argmax()]


"""res = []
for i in range(1,9):
	if i in [5]:
		continue
	source = "Audio/Oded_Kaplan/NL/NL-0"+str(i)+".wav"
	getFFT(source, savepath = "Audio/Oded_Kaplan/NL/NL-0"+str(i)+".eps")
	res.append((i, getHighestFreq(source)))
print(res)"""

x = wavfile.read("Audio/Oded_Kaplan/NL/NL-01.wav")[1][:, 0]
def autocorrelation(x):
    xp = np.fft.ifftshift((x - np.average(x))/np.std(x))
    n, = xp.shape
    xp = np.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
    f = np.fft.fft(xp)
    p = np.absolute(f)**2
    pi = np.fft.ifft(p)
    return np.real(pi)[:n//2]/(np.arange(n//2)[::-1]+n//2)

plt.figure(dpi = 250, figsize = (20,5))
plt.plot(autocorrelation(x)[:8000], '.')
plt.savefig("Audio/Oded_Kaplan/NL/NL-01_autocorrelate.eps")
#print(getHighestFreq("Audio/Oded_Kaplan/PT/PT-01.wav", save = True))

#cutVideo(("Videos/Oded_Kaplan/PT/PT-21.mov", 0.7))
