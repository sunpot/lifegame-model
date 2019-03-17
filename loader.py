import glob
import numpy as np
import lifegame
import cv2


height = 32
width = 48


def load_all():
    input_len = 20
    data_input = []
    data_target = []
    items = glob.glob("./Data/*.npy")
    for idx, item in enumerate(items):
        print("Loading %s/%s"%(idx+1,len(items)))
        array = np.load(item)
        arr_shape = array.shape # (32, 48, n)
        for i in range(0, arr_shape[2]-input_len):
            data_input.append(array[:,:,i:i+input_len].reshape((height*width, input_len)))
            data_target.append(array[:,:,i+input_len].flatten())
        if(1000 < len(data_target)):
            break
    result_input = np.array(data_input, dtype=np.uint8)
    result_target = np.array(data_target, dtype=np.uint8)
    return result_input, result_target


def viewer(slice_data):
    print(slice_data)
    img = lifegame.to_image(slice_data)
    cv2.imshow("test", img)
    cv2.waitKey(0)

def main():
    X, Y = load_all()
    viewer(X[1, :, 4].reshape((height, width)))


if __name__ == "__main__":
    main()