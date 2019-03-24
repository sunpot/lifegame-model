import glob
import numpy as np
import lifegame
import cv2
from sklearn.model_selection import train_test_split
import random


height = 32
width = 48


def load(n=10000, length=20):
    data_input = []
    data_target = []
    items = glob.glob("./Data/*.npy")
    for idx, item in enumerate(items):
        print("Loading %s/%s" % (idx+1, len(items)))
        array = np.load(item)
        arr_shape = array.shape # (32, 48, n)
        for i in range(0, arr_shape[2]-length):
            data_input.append(array[:, :, i:i+length].reshape((height*width, length)))
            data_target.append(array[:, :, i+length].flatten())
        if n < len(data_target):
            break
    result_input = np.array(data_input, dtype=np.uint8).transpose(0, 2, 1)
    result_target = np.array(data_target, dtype=np.uint8)
    return result_input, result_target


def load_one(length=20):
    items = glob.glob("./Val/*.npy")
    idx = random.randint(0, len(items)-1)
    print("Length: %s, Selected index: %s, File path: %s" % (len(items), idx, items[idx]))
    array = np.load(items[idx])
    arr_shape = array.shape # (32, 48, n)
    d_input = []
    d_target = []

    idx = random.randint(0, arr_shape[2]-length-1)
    d_input.append(array[:, :, idx:idx+length].reshape((height*width, length)).T)
    d_target.append(array[:, :, idx+length].flatten())
    print("Data length: %s, Selected index: %s" % (arr_shape[2], idx))
    return np.array(d_input), np.array(d_target)


def diff(data1, data2):
    return np.sum((data1 == data2) == False)


def split(d_input, d_target):
    x, val_x, y, val_y = train_test_split(d_input, d_target, test_size=int(len(d_input) * 0.2), shuffle=False)
    return x, val_x, y, val_y

def viewer(slice_data):
    img = lifegame.to_image(slice_data)
    print("Press any key within activated opencv window.")
    cv2.imshow("test", img)
    cv2.waitKey(0)

def main():
    X, Y = load()
    print("Input shape: %s, Target shape: %s" % (X.shape, Y.shape))


if __name__ == "__main__":
    main()
