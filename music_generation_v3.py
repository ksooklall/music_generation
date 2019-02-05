import pandas as pd
import numpy as np

from util import normalize

from keras import backend as K
from util import get_audio
from keras.layers import Input, Dense, Dropout, Lambda 
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.losses import binary_crossentropy, mse

song = '/home/ksooklall/Music/wav/intro_x.wav'


def preprocessing(X, shape=[96, 16]):
	shape_prod = np.prod(shape)
	if X.shape[0] % np.prod(shape) == 0:
		return X.reshape(shape), shape[1:]
	else:
		# Slice X so it can be reshaped
		idx = int(np.floor(X.shape[0]/shape_prod) * shape_prod)
		return X[:idx].reshape([-1] +shape), shape


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae(X):
	inputs = Input(shape=input_shape, name='encoder_input')
	x = Dense(intermediate_dim, activation='relu')(inputs)
	z_mean = Dense(latent_dim, name='z_mean')(x)
	z_log_var = Dense(latent_dim, kernel_initializer='TruncatedNormal', name='z_log_var')(x)

	# use reparameterization trick to push the sampling out as input
	z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

	# instantiate encoder model
	encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
	encoder.summary()
	#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

	# build decoder model
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	x = Dense(intermediate_dim, activation='relu')(latent_inputs)
	outputs = Dense(original_dim, activation='sigmoid')(x)

	# instantiate decoder model
	decoder = Model(latent_inputs, outputs, name='decoder')
	decoder.summary()
	#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

	# instantiate VAE model
	outputs = decoder(encoder(inputs)[2])
	vae = Model(inputs, outputs, name='vae_mlp')

	return vae, encoder, decoder, z_log_var, z_mean, outputs, inputs


def vae_loss(input_img, output):
	return total_loss


# Preprocessing parameters
rate, data = get_audio(song)

# network parameters
original_dim = 337
input_shape = (original_dim, )
intermediate_dim = 128
batch_size = 64
latent_dim = 32
epochs = 100
learning_rate = 0.001

X = normalize(data)
X = X.reshape(16751, original_dim)
model, encoder, decoder, z_log_var, z_mean, outputs, inputs = vae(X)

# compute the average MSE error, then scale it up, ie. simply sum on all axes
#reconstruction_loss = binary_crossentropy(inputs, outputs) #starting loss 200
reconstruction_loss = mse(inputs, outputs) #starting loss 40
reconstruction_loss *= original_dim

# compute the KL loss
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.square(K.exp(z_log_var)), axis=-1)
# return the average loss over all images in batch
vae_loss = K.mean(reconstruction_loss + kl_loss)    

model.add_loss(vae_loss)
optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=optimizer)

checkpoint = ModelCheckpoint(filepath='x_best_weight.hdf5', verbose=1, save_best_only=True)
history = model.fit(X, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint])
import pdb; pdb.set_trace()
