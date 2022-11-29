'''
    "Life is short, you need Python." 
                                —— Bruce Eckel

    Parallel computing for Landsat time series.

    Design:      Yuan Tao (yuantaogiser@gmail.com);
    Update time: 10/October/2022
'''
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
from pyrsgis import raster
from tqdm import tqdm
from numba import jit
import numpy as np
import itertools
import math
import os

def cal_index(image):
    # image -> type("numpy array"), six bands.

    # To avoid some null values in the image affecting the index calculation, 
    # if you want to calculate some indices that need to be normalized, 
    # you need to use Nan instead of those numbers.
    image[image == -9999] = np.nan
    image[image == 0] = np.nan

    blue = image[0, :, :]
    green = image[1, :, :]
    red = image[2, :, :]
    nir = image[3, :, :]
    swir1 = image[4, :, :]
    swir2 = image[5, :, :]

    msavi = ((2 * nir + 1 - np.sqrt(pow((2 * nir + 1), 2) - 8 * (nir - red)) ) / 2)

    # Of course. MNDWI can be instead of awei_sh. 
    # Here I want to use all the bands for calculating.
    awei_sh = (blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2)
    water = np.full((image.shape[1], image.shape[2]), 100.0)

    # Zhang Yang (https://github.com/zhangyang-2907/DetectingUrbanizedArea/)
    water_index = np.where((awei_sh > 0.0) & (swir1 < 0.1))
    
    water[water_index] = 101.0
    msavi[water_index] = np.nan

    return msavi, water

def parallel_read_image(param,path_list):
    # param     -> index of the list.
    # path_list -> type("str"),the input image path list.

    # the "raster" library loading image is little faster than GDAL/Rasterio library.
    _, image_tmp = raster.read(path_list[param], bands='all')

    # No matter how many exponents the function returns, I recommend that you use only 
    # one variable to receive these values, because the parallel approach causes the final 
    # returned value to be transposed.
    indexCollection = cal_index(image_tmp)

    return indexCollection

@jit(nopython=True)
def numba_interp(data):
    # data -> type("numpy array"), the array shape like (n,).

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
    final = np.interp(X, X_0, Y_0)
    return final

@jit(nopython=True)
def SG_filter_numba(data, window_length, polyorder):
    # data          -> type("numpy array"), the array shape like (n,).
    # window_length -> type("int"), the length of the filter window, and it must be a positive odd integer.
    # polyorder     -> type("int"), the order of the polynomial used to fit the samples.
    
    m = int((window_length - 1) / 2)

    X_array = []
    for i in range(window_length):
        arr = []
        for j in range(polyorder):
            X0 = np.power(-m + i, j)
            arr.append(X0)
        X_array.append(arr)

    X_array = np.array(X_array).reshape((window_length, polyorder)) / 1.0

    # In Numba mode, whenever possible, Numpy should be used instead of "pandas" or "list", etc.
    # That can help you accelerate to unexpected speeds.
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
def dtw_dist(X, Y, w):
    # X -> type("numpy array"), the array shape like (m,n).
    # Y -> type("numpy array"), the array shape like (m,n).
    # w -> type("int"), the length of the dtw window. 

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

@jit(nopython=True)
def Dist_loop(reference,denoised_series):
    # reference       -> type("numpy array"), the array shape like (m,n).
    # denoised_series -> type("numpy array"), the array shape like (m,n).

    # It is highly recommended that when writing Numba, the code should be written 
    # as short and scattered as possible, like C/C++. Numba loves loops.

    Dist = np.zeros(reference.shape[0],)

    for i in range(reference.shape[0]):
        r = np.reshape(reference[i,:].copy(),(-1,1))
        t = np.reshape(denoised_series.copy(),(-1,1))
        Dist[i] = dtw_dist(r, t,3)
    return Dist

@jit(nopython=True)
def dtw_class(denoised_series, reference, water_series):
    # water_series    -> type("numpy array"), the array shape like (m,n).
    # denoised_series -> type("numpy array"), the array shape like (m,n).
    # reference       -> type("numpy array"), the array shape like (m,n).

    # if 'water_index' exists for more than three times,it will be consider as water.
    water_len = len(np.argwhere(water_series == 101))
    if water_len >= 3:
        water_index = 1
    else:
        water_index = 0

    Dist = Dist_loop(reference, denoised_series)

    # Get the minimum value of the year and align it with the corresponding feature
    # It is almost impossible to calculate the two minimum values, so use the "0" index directly, 
    # which can avoid some errors of Numba. Note that Numba has some syntax limitations.
    min_index = np.reshape(np.argwhere(Dist == np.min(Dist)).copy(),(-1,))[0]

    if min_index == 0:
        class_index = 1  # Impervious surface

    elif min_index == 1:
        class_index = 2  # Vegetation

    # Obviously, more features are needed to mitigate soil disturbance.
    # If you care more about bare soil, logical reasoning and more indices are all you need.
    else:
        class_index = 3  # Soil
 
    return np.array([class_index, water_index])

@jit(nopython=True)
def landcover_classification(reference, denoised, water_series,decision_tree):
    # decision_tree    -> type("Boolean"), False or True.

    denoised_len = denoised.shape[1]

    Class = np.zeros((denoised_len,2))

    # year-by-year dtw classification
    for i in range(denoised_len):
        Class[i] = dtw_class(denoised[:,i],reference,water_series[:,i])

    class_series = Class[:,0]
    water_index = Class[:,1]

    if decision_tree:

        # you can defined your own multi-level decision tree.
        print(" Hello world ")

    # "4" means water
    class_series[water_index == 1] = 4

    return class_series

def main(param,reference,time_series):

    tmp_msavi = time_series[:,0,param[0], param[1]]
    tmp_water = time_series[:,1,param[0], param[1]]

    # If the pixel series is all Nan, return it directly
    if np.all(np.isnan(tmp_msavi)):
        return tmp_msavi[0:int(len(tmp_msavi)/12)]

    tmp_msavi = numba_interp(tmp_msavi)

    msavi_denoised = np.reshape(SG_filter_numba(tmp_msavi,5,3).copy(), (12, -1))

    # In Numba mode, reshape any array, "np.reshape(A,shape)" should be used , not "A.reshape(shape)" yet.
    water = np.reshape(tmp_water.copy(), (12, -1))

    tmp_class_series = landcover_classification(reference,msavi_denoised,water,decision_tree=False)

    return tmp_class_series


if __name__=="__main__":

    image_path = r'E:\dataSet\beijing_data\beijing\grid'

    years = range(2012, 2022)
    months = range(1, 13)

    path_to_data = []
    for year in years:
        for month in months:
            data_name = str(year) + '_' + str(month) + '.tif'
            path_to_data_tmp = os.path.join(image_path, data_name)
            path_to_data.append(path_to_data_tmp)

    num_cores = os.cpu_count()
    
    '''
        Note: "backend" has three model("threading","multiprocessing","locky").
              "n_job" means the number of CPU cores that need to work.
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    '''
    # "threading" means multi-threading, and it is good at processing I/O intensive tasks. 
    tmp_time_series = Parallel(n_jobs=num_cores, backend='threading') \
        (delayed(parallel_read_image)(param, path_to_data) for param in tqdm(range(len(path_to_data))))
    time_series = np.array(tmp_time_series)

    print('the shape of Landsat indices time series: ',time_series.shape)

    '''
        Simulation of two reference series representing impervious surface and vegetation.
        Simulation of MSAVI series by attaching harmonics of annual and semi-annual periods.
        PS: Make multiple reference series of real landcover types, and you can get very surprising results.
    '''
    x = np.arange(12)
    a1,a2,a3,a4,a5,a6 = 0.17596,7.81992E-4,-0.12018,-0.1076,-0.00799,0.04429
    veg_ref = a1+a2*x+a3*np.cos(2*math.pi*x/12)+a4*np.sin(2*math.pi*x/12)+a5*np.cos(4*math.pi*x/12)+a6*np.sin(4*math.pi*x/12)
    a1,a2,a3,a4,a5,a6 = 0.06482,1.67557E-4,-0.04148,-0.03031,0.00108,0.00868
    isa_ref = a1+a2*x+a3*np.cos(2*math.pi*x/12)+a4*np.sin(2*math.pi*x/12)+a5*np.cos(4*math.pi*x/12)+a6*np.sin(4*math.pi*x/12)
    references = np.stack((veg_ref,isa_ref))

    # Numbering time series
    i = range(0, time_series.shape[2])
    j = range(0, time_series.shape[3])
    paramList = np.array(list(itertools.product(i, j)))

    # Parallel computing for landcover classification with Numba acceleration.
    tmp_result = Parallel(n_jobs=num_cores,backend='multiprocessing')\
        (delayed(main)(param, references, time_series) for param in tqdm(paramList))

    result = np.array(tmp_result).reshape(time_series.shape[2],time_series.shape[3],int(time_series.shape[0] / 12))

    plt.figure(1)
    plt.imshow(result[:,:, 0])
    plt.figure(2)
    plt.imshow(result[:, :, -1])
    plt.show()

    # If you want to save the final result, change the "False" to "True".
    save_output = False

    if save_output:

        ds_tmp, _ = raster.read(path_to_data[-1], bands='all')

        path_to_output_data = "You need to fill a path."
    
        for i in range(0,result.shape[2]):

            output_data_name = str(years[i]) + '.tif'
            path_to_output = os.path.join(path_to_output_data, output_data_name)

            if os.path.exists(path_to_output):
                os.remove(path_to_output)
    
        for i in range(0,result.shape[2]):

            output_data_name = str(years[i]) + '.tif'
            path_to_output = os.path.join(path_to_output_data, output_data_name)

            raster.export(result[:,:,i], ds_tmp, filename=path_to_output, dtype='float')