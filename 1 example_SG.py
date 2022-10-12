'''
    S-G filtering
    Numba vs. Scipy

    Design:      Yuan Tao (yuantaogiser@gmail.com);
    Update time: 10/October/2022
'''

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from numba import jit
import time
import numpy as np

def SG_filter_python(data, window_length, polyorder):
    '''
    :param data: Array for filtering
    :param window_length: Int
    :param polyorder: Int
    :return: denoised series
    '''

    m = int((window_length - 1) / 2)

    X_array = []
    for i in range(window_length):
        arr = []
        for j in range(polyorder):
            X0 = np.power(-m + i, j)
            arr.append(X0)
        X_array.append(arr)

    X_array = np.array(X_array).reshape((window_length, polyorder)) / 1.0

    B = np.dot(np.dot(X_array, np.linalg.pinv(np.dot(np.transpose(X_array), X_array))), X_array.T)

    data = np.append(np.repeat(data[0], m), data)
    data = np.append(data, np.repeat(data[-1], m))

    B_m = B[m]

    y_array = []
    for n in range(m, data.shape[0] - m):
        y_true = data[n - m: n + m + 1]
        y_filter = np.dot(B_m, y_true)
        y_array.append(float(y_filter))
    return np.array(y_array)

@jit(nopython=True)
def SG_filter_numba(data, window_length, polyorder):

    m = int((window_length - 1) / 2)

    X_array = []
    for i in range(window_length):
        arr = []
        for j in range(polyorder):
            X0 = np.power(-m + i, j)
            arr.append(X0)
        X_array.append(arr)

    X_array = np.array(X_array).reshape((window_length, polyorder)) / 1.0

    B = np.dot(np.dot(X_array, np.linalg.pinv(np.dot(np.transpose(X_array), X_array))), X_array.T)

    data = np.append(np.repeat(data[0], m), data)
    data = np.append(data, np.repeat(data[-1], m))

    B_m = B[m]

    y_array = []
    for n in range(m, data.shape[0] - m):
        y_true = data[n - m: n + m + 1]
        y_filter = np.dot(B_m, y_true)
        y_array.append(float(y_filter))
    return np.array(y_array)


if __name__=="__main__":

    index_x = np.arange(360)

    x = 0.25 * np.sin(0.53*index_x-1.8) + 0.25

    x[10] = x[10] + 1
    x[30] = x[30] + 1
    x[90] = x[90] - 1

    denoise = SG_filter_python(x.reshape(-1,1),5,3)

    start1 = time.time()
    for _ in range(10_000):
        tmp = SG_filter_python(x,5,3)
    time1 = time.time()-start1
    print('python cost time is',time1)

    start2 = time.time()
    for _ in range(100_000):
        tmp = savgol_filter(x, 5, 3)
    time2 = time.time() - start2
    print('scipy cost time is', time2/10)

    start3 = time.time()
    for _ in range(1000_000):
        tmp = SG_filter_numba(x, 5, 3)
    time3 = time.time() - start3
    print('numba cost time is' ,time3/100)

    print('numba is faster than scipy:', time2/10 / (time3 / 100))
    print('numba is faster than python:', time1 / (time3/100))
