import plaidml.keras
plaidml.keras.install_backend()

from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation
# from livelossplot import PlotLossesKeras
import loader


class Params:
    n_in = 1536
    n_hidden = 1000
    n_out = 1536
    epochs = 50
    batch_size = 1000
    inputlen = 20
    test_ratio = 0.2
    lr = 1.0
    n_samples = 40000

    def print(self):
        print("@@@@@@@@@@@@@@@@@@@@@@@@\n")
        print("Hidden layers: %s" % self.n_hidden)
        print("Test ratio: %s" % self.test_ratio)
        print("lr: %s" % self.lr)
        print("N of samples: %s" % self.n_samples)
        print("\n@@@@@@@@@@@@@@@@@@@@@@@@")

p = Params()

d_input, d_target = loader.load(p.n_samples)
x, val_x, y, val_y = loader.split(d_input, d_target, p.test_ratio)
p.print()
model = Sequential()
model.add(LSTM(p.n_hidden, input_shape=(p.inputlen, p.n_in), kernel_initializer='random_normal'))
model.add(Dense(p.n_out, kernel_initializer='random_normal'))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=p.lr, beta_1=0.9, beta_2=0.999))
model.fit(x, y, batch_size=p.batch_size, epochs=p.epochs, validation_data=(val_x, val_y))
