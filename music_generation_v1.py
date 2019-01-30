import os
import time
import sys
import numpy as np

from scipy.io.wavfile import read, write
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam



def get_discriminative_model(input_size):
    model = Sequential()
    model.add(Dense(input_size, activation='relu', input_dim=input_size))
    model.add(Dense(1, activation='sigmoid'))
    return model


def get_generative_model(output_size):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=100))
    model.add(Dense(output_size, activation='tanh'))
    return model


def get_generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def normalize(X):
	return (X - X.min()) / (X.max() - X.min())


def get_audio(fn):
	sr, sd = read(fn)
	sd  = normalize(sd)
	return sr, sd


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


def get_noise(n):
	return np.random.uniform(0, 1, (n, 100))

if __name__ == '__main__':
	epochs = 10
	batch_size = 200
	f_shift = 100
	f_size = 4000
	n_audios_to_dump = 10
	model_dumping_freq = 5
	fn ='/home/ksooklall/Music/wav/intro_x.wav' 
	model_path = '/home/ksooklall/workspace/music_generation/saved_models'
	sr, X_train = get_training_data(fn, f_size, f_shift) #X_train 10000 x 4000
	generator = get_generative_model(f_size)
	discriminator = get_discriminative_model(f_size)
	generator_containing_discriminator = get_generator_containing_discriminator(generator, discriminator)
	
	d_optim = Adam()
	g_optim = Adam()

	generator.compile(loss='binary_crossentropy', optimizer='adam')
	generator_containing_discriminator.compile(loss='binary_crossentropy', optimizer=g_optim)

	discriminator.trainable = True
	discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

	n_minibatches = int(X_train.shape[0]/batch_size) # 50

	for i in range(epochs):
		print('Starting epoch:\t{}'.format(i))
		losses = {'d_losses': [], 'g_losses': []}
		for index in range(n_minibatches):
			noise = get_noise(batch_size)
			generated_audio = generator.predict(noise)
			audio_batch = X_train[index * batch_size: (index + 1) * batch_size]
			X = np.concatenate((audio_batch, generated_audio))
			y = [1] * batch_size + [0] * batch_size
			d_loss = discriminator.train_on_batch(X, y)
			losses['d_losses'].append(d_loss)
			discriminator.trainable = False
			g_loss = generator_containing_discriminator.train_on_batch(noise, [1] * batch_size)
			losses['g_losses'].append(g_loss)
			discriminator.trainable = True
			sys.stdout.write('minibatch: {}/{}\r'.format(index+1, n_minibatches))
			sys.stdout.flush()
		mean_dloss = round(np.mean(losses['d_losses']), 2)
		mean_gloss = round(np.mean(losses['g_losses']), 2)
		print('\n d_loss:{}\tg_loss:{}'.format(mean_dloss, mean_gloss))
		if (i > 0) and ((i+1) % model_dumping_freq ==0):
			timestamp = str(int(time.time()))
			gen_model_savepath = '{}/gen_{}_{}_{}.h5'.format(model_path, timestamp, mean_dloss, mean_gloss)
			dis_model_savepath = '{}/dis_{}_{}_{}.h5'.format(model_path, timestamp, mean_dloss, mean_gloss)
			print('saving models: {}\t{}'.format(gen_model_savepath, dis_model_savepath))
			generator.save(gen_model_savepath)
			discriminator.save(dis_model_savepath)
			print('generating audio samples')
			gend_audio_dirpath = os.path.join('generated_audios', timestamp)
			os.makedirs(gend_audio_dirpath)
			counter = 0
			while counter < n_audios_to_dump:
				noise = get_noise(1)
				gend_audio = generator.predict(noise)
				dd = discriminator.predict(gend_audio)[0] 
				print(dd)
				#import pdb; pdb.set_trace()
				if dd > 0.01:
					print('Counter:\t{}'.format(counter))
					gend_audio = gend_audio[0]
					gend_audio *= 2**15
					outfile = str(counter)+'.wav'
					outfilepath = os.path.join(gend_audio_dirpath, outfile)
					print('{}'.format(outfilepath))
					write(outfilepath, sr, gend_audio.astype(np.int16))
					counter += 1
