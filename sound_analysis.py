import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.io.wavfile import read, write

n =2
samples = ['beat_any_escape_room-10_proven_tricks_and_tips-code', 'intro_x', 'taki_taki']

rate, data = read('/home/ksooklall/Music/wav/{}.wav'.format(samples[n]))
import pdb; pdb.set_trace() 

def plot_wav():
	frequencies, times, spectrogram = signal.spectrogram(data, rate)
	#plt.pcolormesh(times, frequencies, spectrogram)
	#plt.imshow(spectrogram)
	plt.plot(data)
	plt.ylabel('Frequency [Hz]')
	plt.xlabel('Time [sec]')
	plt.show()

def write_wav(t, r, d):
	write(t, r, d)
	return 0
