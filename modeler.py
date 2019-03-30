import plaidml.keras
plaidml.keras.install_backend()

from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation
# from livelossplot import PlotLossesKeras
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import loader


# 次やるやつ→https://omedstu.jimdo.com/2018/06/29/kerasのconvlstmの実装を見る/

class Params:
    n_in = 1600
    n_hidden = 1000
    n_out = 1600
    epochs = 50
    batch_size = 1000
    inputlen = 20
    test_ratio = 0.2
    val_split = 0.1
    lr = 1.0
    n_samples = 5000

    def print(self):
        print("@@@@@@@@@@@@@@@@@@@@@@@@\n")
        print("Hidden layers: %s" % self.n_hidden)
        print("Test ratio: %s" % self.test_ratio)
        print("lr: %s" % self.lr)
        print("N of samples: %s" % self.n_samples)
        print("\n@@@@@@@@@@@@@@@@@@@@@@@@")


p = Params()

d_input, d_target = loader.load_shift(p.n_samples)
# x, y = d_input, d_target
print("%s, %s" % (d_input.shape, d_target.shape))
p.print()

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(20, 40, 40, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')

seq.fit(d_input, d_target, batch_size=p.batch_size,
        epochs=p.epochs, validation_split=p.val_split)

# model = Sequential()
# model.add(LSTM(p.n_hidden, input_shape=(p.inputlen, p.n_in), kernel_initializer='random_normal'))
# model.add(Dense(p.n_out, kernel_initializer='random_normal'))
# model.add(Activation('linear'))
# model.compile(loss='mean_squared_error', optimizer=Adam(lr=p.lr, beta_1=0.9, beta_2=0.999))
# model.fit(x, y, batch_size=p.batch_size, epochs=p.epochs, validation_data=(val_x, val_y))
