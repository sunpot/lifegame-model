#!/usr/bin/python

from __future__ import print_function
import sys
import numpy as np
from scipy import signal
import cv2
import time
import uuid

height = 32
width = 48
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
    print(value)
    return value


def save_all(list):
    fname = '%s.csv' % str(uuid.uuid4())
    for item in list:
        with open(fname, 'a') as f:
            np.savetxt(f,item, fmt='%d', delimiter=',')

def main():
    p = 0.08
    q = np.array([0, 0, 0, 0, 0])
    results = []
    F = init_state(width, height, init_alive_prob=p)
    ret = 0
    wait = 10
    while True:
        img = to_image(F, scale=10.0)
        cv2.imshow("test", img)
        q = queue(q, calc_weight(F))
        is_same = is_all_same(q)
        results.append(F)
        ret = cv2.waitKey(wait)
        F = next_generation(F)
        if ret == ord('r') or is_same:
            save_all(results)
            results = []
            F = init_state(width, height, init_alive_prob=p)
        if ret == ord('s'):
            wait = min(wait*2, 1000)
        if ret == ord('f'):
            wait = max(wait//2, 10)
        if ret == ord('q') or ret == 27:
            break
        if ret == ord('w'):
            np.savetxt("save.txt", F, "%d")
        if ret == ord('l'):
            if os.path.exists("save.txt"):
                F = np.loadtxt("save.txt")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
