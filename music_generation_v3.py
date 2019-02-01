from keras import backend as K
from util import get_audio
from keras.layers import Input, Dense, Dropout, Lambda 
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import plot_model

song = '/home/ksooklall/Music/wav/intro_x.wav'

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
	z_log_var = Dense(latent_dim, name='z_log_var')(x)

	# use reparameterization trick to push the sampling out as input
	# note that "output_shape" isn't necessary with the TensorFlow backend
	z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

	# instantiate encoder model
	encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
	encoder.summary()
	plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

	# build decoder model
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	x = Dense(intermediate_dim, activation='relu')(latent_inputs)
	outputs = Dense(original_dim, activation='sigmoid')(x)

	# instantiate decoder model
	decoder = Model(latent_inputs, outputs, name='decoder')
	decoder.summary()
	plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

	# instantiate VAE model
	outputs = decoder(encoder(inputs)[2])
	vae = Model(inputs, outputs, name='vae_mlp')

	return vae, encoder, decoder

# network parameters
original_dim = 337
input_shape = (original_dim, )
intermediate_dim = 1024
batch_size = 128
latent_dim = 64
epochs = 10

rate, data = get_audio(song)
# 2393, 337, 7
X = data.reshape(16751, original_dim)
model, encoder, decoder = vae(X)
model.compile(optimizer='adam', loss='binary_crossentropy')

history = model.fit(X, X, batch_size=batch_size, epochs=epochs)
import pdb; pdb.set_trace()
