from matplotlib import pyplot as plt
import parselmouth as pm
import numpy as np
import os

folder = "Audio/Oded_Kaplan/"
files = ["kaplanfalse1.wav", "kaplanfalse2.wav", "kaplantrue1.wav", "kaplantrue2.wav"]

for file in files:
    if not file[-4:] == ".wav":
        continue

    path = folder + file

    sound = pm.Sound(path)
    pitch = sound.to_pitch(time_step = 0.01)
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    pitch_values[pitch_values > 120] = np.nan
    plt.figure(dpi = 250)
    plt.plot(pitch.xs(), pitch_values, 'r.')
    plt.savefig(path[:-4]+"_pitch.eps")
    plt.close()
    print(file, np.nanmean(pitch_values))
