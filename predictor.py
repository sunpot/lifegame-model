from keras.models import load_model
import loader


model = load_model('lifegame_2019-03-24.h5')
d_input, d_target = loader.load_one()
loader.diff(d_target, d_target)