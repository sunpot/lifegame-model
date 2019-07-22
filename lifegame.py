#!/usr/bin/python

from __future__ import print_function
import sys
import numpy as np
from scipy import signal
import cv2
import time
import uuid
from io import BytesIO
import bz2

height = 64
width = 64
mask = np.ones((3, 3), dtype=int)


def init_state(width, height, init_alive_prob=0.5):
    N = width*height
    v = np.array(np.random.rand(N) + init_alive_prob, dtype=np.uint8)
    return v.reshape(height, width)


def count_neighbor(F):
    return signal.correlate2d(F, mask, mode="same", boundary="wrap")


def next_generation(F):
    N = count_neighbor(F)
    G = (N == 3) + F * (N == 4)
    return G


def to_image(F, scale=3.0):
    img = np.array(F, dtype=np.uint8)*255
    W = int(F.shape[1]*scale)
    H = int(F.shape[0]*scale)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return img


def queue(src, a):
    dst = np.roll(src, -1)
    dst[-1] = a
    return dst


def is_all_same(a):
    b = a[::2]
    return not np.any((b == b[-1]) == False)


def calc_weight(F):
    base_array = np.arange(0, height*width).reshape(height, width)
    tmp = base_array * F
    value = tmp.sum()
    sys.stdout.write("%s,"%value)
    return value


def save_all(list):
    fname = './Data/%s.npy' % str(uuid.uuid4())
    if len(list) < 20:
        print("\n\nDiscard.\n\n")
        return
    print("\n\nSaving %s\n\n"%fname)
    array = np.zeros((len(list),height, width))
    for i in range(len(list)):
        array[i,:,:] = list[i]
    np.save(fname, array)
    return


def hasCycle(arr):
    """
    Floyd's cycle-finding algorithm is a pointer algorithm that uses 2 pointers,
    which move through the sequence at different speeds.
    """
    osoi = np.array(arr)
    hayai = np.array(arr)

    i_osoi = 0
    i_hayai = 0

    while True:
        i_osoi += 1
        i_hayai += 2
        if len(arr) <= i_osoi or len(arr) <= i_hayai:
            break
        try:
            if osoi[i_osoi] == hayai[i_hayai]:
                print("Cycle found")
                return True
        except IndexError:
            return False
    return False

def main():
    p = 0.08
    q = np.array([0, 0, 0, 0, 0])
    results = []
    weights = []
    F = init_state(width, height, init_alive_prob=p)
    ret = 0
    wait = 10
    i = 0
    while True:
        # if 0 == i%50:
        #     img = to_image(F, scale=10.0)
        #     cv2.imshow("test", img)
        #     i = 0
        # i += 1
        weight = calc_weight(F)
        weights.append(weight)
        q = queue(q, weight)
        is_same = is_all_same(q)
        results.append(F)

        F = next_generation(F)
        if is_same or hasCycle(weights):
            save_all(results)
            results = []
            weights = []
            F = init_state(width, height, init_alive_prob=p)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
