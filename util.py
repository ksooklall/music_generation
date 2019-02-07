import numpy as np
from scipy.io.wavfile import read, write
from pydub import AudioSegment

def normalize(X):
	"""
	Normalize values between [0, 1)
	"""
	X_max = float(X.max())
	X_min = float(X.min())
	norm = (X - X_min) / (X_max - X_min)
	return norm, X_max, X_min


def denormalize(n, X_max, X_min):
	"""
	Denormalizes X to not be normal
	"""
	return n * (X_max - X_min) + X_min

def get_audio(filename):
	"""
	read audio file and return normalized
	"""
	rate, data = read(filename)
	return rate, data


def set_audio(filename, rate, data):
	"""
	Write data into wav file
	Arguments:
		filename (str)
		rate (int)
		data (np.array)
	"""
	write(filename, rate, data)
	return 0

def get_training_data(fn, f_size, f_shift):
	# f_size -> batch_size
	sr, sd = get_audio(fn)
	X_train = []
	base = 0
	n = int((len(sd) - f_size) / float(f_shift))
	while len(X_train) < 10000:
			X_train.append(sd[base: base+f_size])
			base += f_shift
	X_train = np.array(X_train)
	return sr, X_train


def convert_mp3_to_wav(mp3_file):
	"""
	Given path to mp3 file, converts to python wav file
	"""
	sound = AudioSegment.from_mp3(mp3_file)
	sound.export('{}.wav'.format(mp3_file.split('.')[-1]), sound)
	return 0
