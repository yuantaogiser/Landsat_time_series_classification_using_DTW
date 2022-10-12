'''
    Parallel computing
    Just for describing how the parallel happens

    Parallel library:
        joblib 0.17.0 (https://joblib.readthedocs.io/en/latest/)

    Design:      Yuan Tao (yuantaogiser@gmail.com);
    Update time: 10/October/2022
'''
from joblib import Parallel,delayed

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
import time
import os

def main(x,y):
    dist_fastdtw, _ = fastdtw(x, y, radius=3, dist=euclidean)
    return dist_fastdtw

if __name__=="__main__":

    index_x = np.linspace(0, 23, 24)
    x = 0.25 * np.sin(0.53 * index_x - 1.8) + 0.25
    y = 0.25 * np.sin(0.53 * index_x - 0.8) + 0.25

    start1 = time.time()
    for _ in range(10000):
        dist_fastdtw, _ = fastdtw(x, y, radius=3, dist=euclidean)
    print('non-parallel fastdtw cost time is:', time.time() - start1)

    start2 = time.time()
    num_cpu = os.cpu_count() # Note: this will call all your logic cores.
    Parallel(n_jobs=num_cpu,backend='multiprocessing')(delayed(main)(x,y) for _ in range(10000))
    print('parallel fastdtw cost time is:',time.time()-start2)
