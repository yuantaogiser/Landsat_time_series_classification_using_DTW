'''
    linear interpolation
    Numba vs. Scipy

    Design:      Yuan Tao (yuantaogiser@gmail.com);
    Update time: 10/October/2022
'''

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import time

@jit(nopython=True)
def numba_interp(data):
    
    # 1.5*IQRs are used for removing the gross error
    pixel_timeseries = data.copy()
    Q1 = np.nanpercentile(pixel_timeseries, 25)
    Q3 = np.nanpercentile(pixel_timeseries, 75)
    IQR = Q3 - Q1
    Q_max = Q3 + IQR * 1.5
    Q_min = Q1 - IQR * 1.5
    pixel_timeseries[pixel_timeseries > Q_max] = np.nan
    pixel_timeseries[pixel_timeseries < Q_min] = np.nan

    data = pixel_timeseries.copy()

    X = np.arange(0,len(data))
    # Fill in the gaps at the beginning and end based on nearest neighbor interpolation
    ValidDataIndex = X[np.where(np.isnan(data) == 0)]
    if ValidDataIndex[-1] < len(data) - 1:
        data[ValidDataIndex[-1] + 1:] = data[ValidDataIndex[-1]]
    if ValidDataIndex[0] >= 1:
        data[:ValidDataIndex[0]] = data[ValidDataIndex[0]]

    Y_0 = data[np.where(np.isnan(data) != 1)]
    X_0 = X[np.where(np.isnan(data) != 1)]

    # interpolate
    final = np.interp(X, X_0, Y_0)

    return final

def scipy_interp(data):

    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    Q_max = Q3 + IQR * 1.5
    Q_min = Q1 - IQR * 1.5
    data[data > Q_max] = np.nan
    data[data < Q_min] = np.nan

    X = np.arange(len(data))
    ValidDataIndex = X[np.where(np.isnan(data) == 0)]
    if ValidDataIndex[-1] < len(data) - 1:
        data[ValidDataIndex[-1] + 1:] = data[ValidDataIndex[-1]]
    if ValidDataIndex[0] >= 1:
        data[:ValidDataIndex[0]] = data[ValidDataIndex[0]]

    Y_0 = data[np.where(np.isnan(data) != 1)]
    X_0 = X[np.where(np.isnan(data) != 1)]
    IRFunction = interp1d(X_0, Y_0, kind='linear')
    Fill_X = X[np.where(np.isnan(data) == 1)]
    Fill_Y = IRFunction(Fill_X)
    data[Fill_X] = Fill_Y
    return data

if __name__=="__main__":

    index_x = np.arange(360)

    x = 0.25 * np.sin(0.53*index_x-1.8) + 0.25

    x[10:12],x[30],x[90],x[92],x[100:102],x[120:122] = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan

    start1 = time.time()
    for _ in range(100_000):
        data = x.copy()
        tmp = scipy_interp(data)
    print('scipy interpolate cost time is:',time.time()-start1)

    start2 = time.time()
    for _ in range(1000_000):
        data = x.copy()
        tmp = numba_interp(data)
    print('numba interpolate cost time is:',(time.time() - start2)/10)

    # plt.figure()
    # plt.plot(scipy_interp(x.copy()))
    # plt.plot(numpy_interp(x.copy()))
    # plt.plot(x)
    #
    # plt.figure(2)
    # plt.plot(numpy_interp(x.copy())-scipy_interp(x.copy()))
    #
    # plt.show()