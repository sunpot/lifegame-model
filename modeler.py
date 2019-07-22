import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation
# from livelossplot.keras import PlotLossesCallback
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from datetime import datetime as dt

height = 64
width = 64

class Params:
    n_in = height * width
    n_hidden = 5000
    n_out = height * width
    epochs = 20
    batch_size = 10
    inputlen = 5
    test_ratio = 0.2
    val_split = 0.1
    lr = 0.1
    n_samples = 1000

    def print(self):
        print("@@@@@@@@@@@@@@@@@@@@@@@@\n")
        print("Hidden layers: %s" % self.n_hidden)
        print("Test ratio: %s" % self.test_ratio)
        print("lr: %s" % self.lr)
        print("N of samples: %s" % self.n_samples)
        print("\n@@@@@@@@@@@@@@@@@@@@@@@@")


def load(n=10000):
    input_len = 20
    data_input = []
    data_target = []
    items = glob.glob("./Data/*.npy")
    for idx, item in enumerate(items):
        # print("Loading %s/%s" % (idx+1, len(items)))
        array = np.load(item)
        arr_shape = array.shape # (height, width, n)
        break_flg = False
        for i in range(0, arr_shape[2]-input_len):
            data_input.append(array[:, :, i:i+input_len].reshape((height*width, input_len)))
            data_target.append(array[:, :, i+input_len].flatten())
            if n == len(data_target):
                break_flg = True
                break
        if break_flg:
            break
    result_input = np.array(data_input, dtype=np.uint8).transpose(0, 2, 1)
    result_target = np.array(data_target, dtype=np.uint8)
    return result_input, result_target


def load_shift(n=10000, length=20):
    input_len = length
    data_input = []
    data_target = []
    items = glob.glob("./Data/*.npy")
    for idx, item in enumerate(items):
        # print("Loading %s/%s" % (idx+1, len(items)))
        array = np.load(item)
        #print(array.shape)
        arr_shape = array.shape # (32, 48, n)
        break_flg = False
        for i in range(0, arr_shape[0]-input_len):
            i2 = i + 1
            data_input.append(array[i:i+input_len,:, :])
            data_target.append(array[i2:i2+input_len, :, :])
            if n == len(data_target):
                break_flg = True
                break
        if break_flg:
            break
    result_input = np.array(data_input, dtype=np.uint8).reshape((n, input_len, height, width, 1)) #.transpose(0, 3, 1, 2).reshape((n, input_len, height, width, 1))
    result_target = np.array(data_target, dtype=np.uint8).reshape((n, input_len, height, width, 1)) #.transpose(0, 3, 1, 2).reshape((n, input_len, height, width, 1))
    return result_input, result_target
  
def split(d_input, d_target, test_ratio=0.2):
    x, val_x, y, val_y = train_test_split(d_input, d_target, test_size=int(len(d_input) * test_ratio), shuffle=False)
    return x, val_x, y, val_y

# Initialize
p = Params()
cb_es = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
# cb_plot = PlotLossesCallback()

# Prepare dataset
d_input, d_target = load_shift(n=p.n_samples, length=p.inputlen)
# x, y = d_input, d_target
print("%s, %s" % (d_input.shape, d_target.shape))
p.print()

# Fit
seq = Sequential()
seq.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
                   input_shape=(p.inputlen, height, width, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(filters=256, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adam')
seq.fit(d_input, d_target, batch_size=p.batch_size,
        epochs=p.epochs, validation_split=p.val_split) #, callbacks=[cb_plot])

# Finalize
tdatetime = dt.now()
fname = "lifegame_%s.h5"%"{0:%Y-%m-%d-%H:%M:%S}".format(tdatetime)
print(fname)
seq.save(fname)
p.print()
# !cp -f ./lifegame_*.h5 /lifegame/My\ Drive/lifegame/