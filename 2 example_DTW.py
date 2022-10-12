'''
    dynamic time warping (DTW)
    Numba vs. Python library

    Design:      Yuan Tao (yuantaogiser@gmail.com);
    Update time: 10/October/2022
'''
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from dtaidistance import dtw

from numba import jit
import numpy as np
import time

@jit(nopython=True)
def dtw_dist(X, Y, w):
    '''
    :param X: 2d-Array
    :param Y: 2d-Array
    :param w: Int
    :return:  dtw distance
    '''

    # Create cost matrix via broadcasting with large int
    n_X, n_Y = X.shape[1], Y.shape[1]
    C = np.full((n_X+1, n_Y+1), np.inf)

    # Initialize the first row and column
    C[0, 0]= 0
    w = max(w, abs(n_X-n_Y))
    for i in range(1, n_X+1):
        X_vec = X[:, i-1]

        # Populate rest of cost matrix within window
        for j in range(max(1, i-w), min(n_Y+1, i+w+1)):

            # Absolute values are used instead of the operation(Sqrt & Square root) because they are equivalent
            diff = np.abs(X_vec-Y[:, j-1])
            cost = np.sum(diff)
            C[i, j] = cost+min(C[i-1, j], C[i, j-1], C[i-1, j-1])

    return C[n_X, n_Y]

def dtw_python(X, Y, w):

    n_frame_X, n_frame_Y = X.shape[1], Y.shape[1]
    D = np.full((n_frame_X+1, n_frame_Y+1), np.inf)

    D[0, 0]= 0
    w = max(w, abs(n_frame_X-n_frame_Y))

    for i in range(1, n_frame_X+1):
        X_vec = X[:, i-1]

        for j in range(max(1, i-w), min(n_frame_Y+1, i+w+1)):

            diff = np.abs(X_vec-Y[:, j-1])

            cost = np.sum(diff)
            D[i, j] = cost+min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    return D[n_frame_X, n_frame_Y]

if __name__=="__main__":

    index_x = np.linspace(0, 23, 24)
    x = 0.25 * np.sin(0.53 * index_x - 1.8) + 0.25
    y = 0.25 * np.sin(0.53 * index_x - 0.8) + 0.25

    start1 = time.time()
    for _ in range(1000):
        dist_python = dtw_python(x.reshape(1,-1),y.reshape(1,-1),3)
    print('python dtw cost time is:',time.time()-start1)

    start2 = time.time()
    for _ in range(1000):
        dist_fastdtw, _ = fastdtw(x, y,radius=3, dist=euclidean)
    print('fastdtw cost time is:',time.time()-start2)

    start3 = time.time()
    for _ in range(1000_000):
        dist_numba = dtw_dist(x.reshape(1,-1), y.reshape(1,-1), 3)
    print('numba dtw cost time is:',(time.time()-start3)/1000)

    start4 = time.time()
    for _ in range(1000_000):
        distance_fast = dtw.distance_fast(x, y, window=3)

    # It is the fastest DTW. but it have different result with other dtw.
    print('distance_fast dtw cost time is:',(time.time()-start4)/1000)


