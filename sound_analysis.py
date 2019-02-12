import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.io.wavfile import read, write
from scipy.signal import spectrogram
import scipy.signal as signal

n =2
samples = ['intro_x', 'taki_taki', 'chris_brown_with_you']
path = '/home/ksooklall/Music/wav/{}.wav'

def plot_wav(song, data, rate):
	f, t, Z = signal.stft(data, rate)
	#plt.imshow(spectrogram)
	print('Song: {}\tShape: {}'.format(song, data.shape))
	#plt.specgram(data.flatten(), cmap='rainbow', Fs=rate)
	#plt.pcolormesh(t, f, np.abs(Z), cmap='rainbow')
	import pdb; pdb.set_trace()
	plt.plot(data.flatten()[:1000])
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.title(song)
	plt.savefig(song +'_raw')
	#plt.show()



def write_wav(t, r, d):
	write(t, r, d)
	return 0


#for i in samples:
	#rate, data = read('/home/ksooklall/Music/wav/{}.wav'.format(i))
	#plot_wav(i, data, rate)

def slice_data(data, pct=1):
	"""
	Samples from data between start and end, or a certain pct starting from the center
	and moving out in a (PCT - 0.5) radius
	"""
	if pct > 1:
		raise "Percent must be less than 1"

	length = data.shape[0]
	start = int(length * (1 - pct))
	end = int(length * pct)
	return data[start:end]

rate, data = read('/home/ksooklall/Music/wav/{}.wav'.format(samples[1]))
import pdb; pdb.set_trace()

df = pd.DataFrame()

