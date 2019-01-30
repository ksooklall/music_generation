import keras
from keras.utils import plot_model

PARAM_SIZE = 120
DROPOUT=0.1
MAX_LENGTH=16

def model(y):
	x_in = Input(shape=y_shape[1:])
	x = Reshape((y_shape[1:], -1))(x_in)
	x = TimeDistributed(Dense(2000, activation='relu'))(x)
	x = TimeDistributed(Dense(200, activation='relu'))(x)
	x = Flatten()(x)
	x = Dense(1600, activation='relu')(x)
	x = Dense(120)(x)
	x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
	x = Dense(1600, name='encoder')(x)
	x = BatchNormalization(momentum=BN_M)(x)
	x = Activation('relu')(x)
	x = Dropout(DROPOUT)(x)

	x = DENSE(MAX_LENGTH * 200)(x)
	x = Reshape((MAX_LENGTH, 200))(x) #193
	x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
	x = Activation('relu')(x)
	x = Dropout(DROPOUT)(x) # 197

	x = TimeDistributed(Dense(2000))(x)
	x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
	x = Activation('relu')(x)
	x = Dropout(DROPOUT)(x) #204

	x = TimeDistributed(Dense(y_shape[2] * y_shape[3], activation='sigmoid'))(x)
	x = Reshape((y_shape[1], y_shape[2], y_shape[3]))(x) #209

	model = Model(x_in, x)

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

model.compile(optimizer=RMSprop(lr=LR), loss='binary_crossentropy')
plot_model(model, to_file='model.png', show_shapes=True)
history = model.fit(y_train, y_train, batch_size=BATCH_SIZE, epochs=1)

