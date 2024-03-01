import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from glob import glob

import librosa
import librosa.display
import IPython.display as ipd

g_signal,g_sr=librosa.load('/content/smooth-ac-guitar-loop-93bpm-137706.mp3')
d_signal,d_sr=librosa.load('/content/amen-break-no-copyright-remake-120bpm-25924.mp3')

print("len of g_signal:{}--signal rate:{}".format(len(g_signal),g_sr))
print("len of d_signal:{}--signal rate:{}".format(len(d_signal),d_sr))

#graph frequency vs time
time = np.arange(len(g_signal))/g_sr
print(time)
plt.plot(time,g_signal)

time = np.arange(len(d_signal))/d_sr
plt.plot(time[100:200],d_signal[100:200])


#transforing time domain to frequency domain(spectrun)
g_fft=np.fft.fft(g_signal)
g_fft_abs=np.abs(g_fft)
d_fft=np.fft.fft(d_signal)
d_fft_abs=np.abs(d_fft)

plt.plot(g_fft_abs)
plt.xlabel("frequency in HZ")
plt.ylabel("Amplitude (Magnitude)")

plt.plot(d_fft_abs)
plt.xlabel("frequency in HZ")
plt.ylabel("Amplitude (Magnitude)")
# Calculate power spectrum
g_frequency=np.linspace(0,g_sr,len(g_signal))
d_frequency=np.linspace(0,d_sr,len(d_signal))

left_frequency=g_frequency[:int(len(g_frequency)/2)]
left_power_spectrum=g_fft_abs[:int(len(g_frequency)/2)]


left_frequency=d_frequency[:int(len(d_frequency)/2)]
left_power_spectrum=d_fft_abs[:int(len(d_frequency)/2)]
plt.plot(left_frequency,left_power_spectrum)

#n_fft--number of samples for each segment
#hop_lenght-- overlapping sample between to consecutive segments
g_stft=librosa.core.stft(g_signal,hop_length=512,n_fft=2048)
d_stft=librosa.core.stft(d_signal,hop_length=512,n_fft=2048)


spectrogram=np.abs(g_stft)
log_spectrogram=librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram,sr=g_sr,hop_length=512)
plt.xlabel('time')
plt.ylabel('frequency')
plt.colorbar()

spectrogram=np.abs(d_stft)
log_spectrogram=librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram,sr=d_sr,hop_length=512)
plt.xlabel('time')
plt.ylabel('frequency')
plt.colorbar()




# Parameters
hop_length = 512  # Number of samples between consecutive frames
n_fft = 2048      # Number of data points used in each short-time segment
n_mfcc = 13       # Number of MFCC coefficients to extract

# Compute MFCCs
MFCCs = librosa.feature.mfcc(y=g_signal,sr=g_sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs,sr=g_sr,hop_length=512)
plt.xlabel('time')
plt.ylabel('mfcc')
plt.colorbar()


MFCCs = librosa.feature.mfcc(y=d_signal,sr=d_sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs,sr=d_sr,hop_length=512)
plt.xlabel('time')
plt.ylabel('mfcc')
plt.colorbar()