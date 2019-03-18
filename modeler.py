from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation
import loader

n_in = 1536
n_hidden = 20
n_out = 1536
epochs = 40
batch_size = 10
inputlen = 20

d_input, d_target = loader.load()
x, val_x, y, val_y = loader.split(d_input, d_target)

model = Sequential()
model.add(SimpleRNN(n_hidden, input_shape=(inputlen, n_in), kernel_initializer='random_normal'))
model.add(Dense(n_out, kernel_initializer='random_normal'))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.1, beta_1=0.9, beta_2=0.999))
model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=(val_x, val_y))
