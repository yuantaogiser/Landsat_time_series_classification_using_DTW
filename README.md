# Landsat time series high-performance classification based on dynamic time warping for landcover mapping
  
    Designed    : Ph.D Student, Yuan Tao [1,2]
    Institution : [1] CUMT (China University of Mining and Technology)
                  [2] NGCC (Natinal Geomatics Center of China)
    Update time : 10, October, 2022
    E-mail      : yuantaogiser@gmail.com

## 1. Introduction
   With the rapid iteration of computer technology, high-performance computing resources are becoming more and more accessible, and even individuals can pay these expensive bills to purchase these high-performance devices. We should make full use of our computing resources to solve some of those unimaginable things, using only local resources. Here we go.
   
## 2. Contents
   ### 2.1 example_SG.py
   We reproduce the SG filter with Numba acceleration. It is very efficient when doing a lot of calculations. As we know, the first call to a Numba compiled function is time consuming. 
   
   ### 2.2 example_DTW.py
   We reproduce the dynamic time warping with Numba acceleration.
   
   ### 2.3 example_interpolation.py
   We reproduce the linear interpolation with Numba acceleration.
   
   ### 2.4 example_parallel.py
   We would like to briefly show here how to perform parallel computing by using joblib library, so that readers can better understand parallel in "DTW_classification.py".
   
   ### **2.5 DTW_classification.py** 
   **We present here the main framework of DTW for long-term landcover classification in parallel computing.** Below We will point out a few points to note. (1) joblib is a lightweight parallel library for a single node(computer). If you want to do distributed parallel computing over a local area network, We would recommend you to use the dask library for distributed processing. (2) We give the main frame of DTW classification, which makes our code more extensible. We have reserved a lot of details for you to extend this framework. Especially for some complex landcover types, it needs to be classified by a specific index and/or a specific decision tree. (3) We strongly recommend users that when using DTW, the reference series must use periodic series, not those that are confusing.

## 3. How to use "DTW_classification.py" ?
   If you just want to run it, run it. But I think you probably need to extend it. **First, the correct image folder path should be filled in correctly. Then, We suggest adding a periodic index according to your own needs to deal with different landcover types. Finally, a decision tree should be built to improve the classification accuracy.**
   
## 4. Outlooks
   GPUs should be used, which may significantly improve computational efficiency. However, when We use the GPU for computing, I found that more time was spent in the communication of the video memory, which led to a decrease in computing efficiency. I hope this wheel can move forward, if anyone implements efficient GPU computing based on Python, please send me an email , I hope to be able to learn such techniques. Also, If you have anything you would like to discuss, please email me.
   
## PS: My environment
    Laptop configuration:
    CPU: i7-12700H
    RAM: 16G

    Main library:
    Python : 3.9.12
    joblib : 0.17.0
    numpy  : 1.21.5
    numba  : 0.55.1
    scipy  : 1.7.3

## What is the flowchart?

### _Tao Y., Liu W., Chen J., et al. (2023) Mapping 30m China annual impervious surface from 1992 to 2021 via a multi-level classification approach. International Journal of Remote Sensing. 44(13):4086-4114. https://doi.org/10.1080/01431161.2023.2232541_


https://www.tandfonline.com/na101/home/literatum/publisher/tandf/journals/content/tres20/2023/tres20.v044.i13/01431161.2023.2232541/20230714/images/large/tres_a_2232541_f0002_c.jpeg
