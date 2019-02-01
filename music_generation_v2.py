import keras
from keras import backend as K
from keras.utils import plot_model
from util import get_audio
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Activation, TimeDistributed, Dropout, Reshape, Flatten

PARAM_SIZE = 120
DROPOUT=0.1
MAX_LENGTH=7
BN_M=0.9


def autoencoder(X):
	net_in = Input(shape=X.shape[1:])
	#net = Reshape((X.shape[1:], -1))(net_in)
	net = Dense(2000, activation='relu')(net_in)
	import pdb; pdb.set_trace()
	net = Dense(200, activation='relu')(net)
	net = Flatten()(net)
	net = Dense(1600, activation='relu')(net)
	net = Dense(120)(net)
	net = BatchNormalization(momentum=BN_M, name='pre_encoder')(net)
	net = Dense(1600, name='encoder')(net)
	net = BatchNormalization(momentum=BN_M)(net)
	net = Activation('relu')(net)
	net = Dropout(DROPOUT)(net)

	net = DENSE(MAX_LENGTH * 200)(net)
	net = Reshape((MAX_LENGTH, 200))(net) #193
	net = BatchNormalization(momentum=BN_M)(net)
	net = Activation('relu')(net)
	net = Dropout(DROPOUT)(net) # 197

	net = Dense(2000)(net)
	net = BatchNormalization(momentum=BN_M)(net)
	net = Activation('relu')(net)
	net = Dropout(DROPOUT)(net) #204

	net = Dense(X.shape[2] * X.shape[3], activation='sigmoid')(net)
	net = Reshape((X.shape[1], X.shape[2], X.shape[3]))(net) #209

	model = Model(net_in, x)

	return model

rate, data = get_audio('/home/ksooklall/Music/wav/intro_x.wav') 
X = data.reshape(-1, MAX_LENGTH)

model = autoencoder(X)
model.compile(optimizer=RMSprop(lr=LR), loss='binary_crossentropy')
plot_model(model, to_file='model.png', show_shapes=True)
history = model.fit(y_train, y_train, batch_size=BATCH_SIZE, epochs=1)

