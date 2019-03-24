from keras.models import load_model
import loader
import numpy as np
import platform
import os

if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = load_model('lifegame_2019-03-24.h5')
d_input, d_target = loader.load_one()

in_ = d_input # x の先頭 (1,20,1) 配列
#predicted = [None for _ in range(inputlen)]
#for _ in range(len(rawdata) - inputlen):
out_ = model.predict(d_input) # 予測した値 out_ は (1,1) 配列
#in_ = np.concatenate( (in_.reshape(inputlen, n_in)[1:], out_), axis=0 ).reshape(1, inputlen, n_in)
#predicted.append(out_.reshape(-1))

#loader.viewer(d_target.reshape(32, 48))
#loader.viewer(out_.reshape(32, 48))

np.savetxt('d_target.csv', d_target.reshape(32, 48), delimiter=',')
np.savetxt('out_.csv', out_.reshape(32, 48), delimiter=',')
#print(out_.reshape(32, 48))
# loader.diff(d_target, d_target)