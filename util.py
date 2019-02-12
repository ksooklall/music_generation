import numpy as np
from functools import reduce
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


def convert_mp3_to_wav(mp3_file):
	"""
	Given path to mp3 file, converts to python wav file
	"""
	sound = AudioSegment.from_mp3(mp3_file)
	sound.export('{}.wav'.format(mp3_file.split('.')[-1]), sound)
	return 0


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


def factors(n):    
	#https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))


def batch_data(data, rows=96, colos=96):
	"""
	Reshapes data [-1, rows, cols]
	"""
	if data.shape[0] % (rows * cols) == 0:
		return data.reshape(-1, rows, cols)
	X = []
	batch = rows * cols
	for i in range(0, len(data)/ batch, batch):
		X.append(data[i: i + batch].reshape(rows, cols))
	return np.array(X)
