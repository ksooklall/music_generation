import numpy as np
from scipy.io.wavfile import read, write

def normalize(X):
	"""
	Normalize values between [0, 1)
	"""
        return (X - X.min()) / (X.max() - X.min())


def get_audio(filename):
	"""
	read audio file and return normalized
	"""
        rate, data = read(filename)
        data  = normalize(data)
        return rate, data


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

