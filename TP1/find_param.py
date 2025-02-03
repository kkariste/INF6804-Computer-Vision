## find_param
# The purpose of this code is to find optimal parameters
# Use the for loops to test different values of different param

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.feature
from matplotlib.pyplot import imread
import cv2
import numpy as np
from pylab import ginput
from scipy import signal
from scipy import stats
import matplotlib
from scipy.ndimage import convolve
from skimage.color import rgb2gray
from skimage import data
import statistics
from methodsProj1 import *
import time

bins = 256
hist_min = 0
numPoints = 20
radius = 3
results = [0] * 8
image_query, image_database,image_query_gray,image_database_gray, names,names_query = readFiles()
textures = ['airplane', 'ball', 'car', 'cat', 'dolphin', 'face', 'lotus', 'pickles']
best_j = 0
best_k = 0
best_i = 0
best_score = 0
data_x = []
data_y = []
time_y = []
dx = [3, 3 , 3]
da = [np.pi/3,5*np.pi/6,5*np.pi/6]
niveaux = [1,2,4,8,16,32,64,128]
levels = niveaux[0]
hist_max = np.round(256/levels)
method_score = [0] * 8
method_names = ["Correlation","Chi-Square","Intersection","Bhattacharyya","Hellinger","Chi-Square Alt","Kullback-Leibler","L2"]
method_used = "LBP"
for i in range (0,1):
        for k in range(0,1):
                for j in range(0,1):
                        ts = time.time()
                        if(method_used == "LBP"):
                                hist_query = getLBPhist(image_query_gray, numPoints, radius)
                                hist_database = getLBPhist(image_database_gray, numPoints, radius)
                                metric = 7
                        else:
                                hist_query = getMChist(image_query,image_query_gray,bins,hist_min,hist_max,dx,da,1,1,8,levels)
                                hist_database = getMChist(image_database,image_database_gray,bins,hist_min,hist_max,dx,da,1,1,8,levels)
                                metric = 3

                        for N in range(0, len(hist_query) - 1):
                                results[N] = findBestMatch(hist_query[N],hist_database,names,names_query[N],metric)

                        score = np.average(results)*100/3

                        data_x.append(i)
                        data_y.append((results))
                        te = time.time()
                        time_y.append(te - ts)
                        if (score > best_score):
                                best_score = score
                                best_j = j
                                best_k = k
                                best_i = i



print(best_score)
print(best_i)
print(best_k)
print(best_j)
print(te-ts)
plt.bar(textures, results)
plt.xlabel('Comparaison method')
plt.ylabel('Score (%)')
plt.title('Best method for LBP')
plt.show()

plt.plot(data_x, data_y,'o')
plt.xlabel('Range from referenced pixel')
plt.ylabel('Score (%)')
plt.title('Score for MC, 45 degrees')
plt.show()

plt.plot(data_x, time_y,'o')
plt.xlabel('Range from referenced pixel')
plt.ylabel('Execution time (s)')
plt.title('Time for MC, 45 degrees')
plt.show()

